import numpy as np
import scipy.sparse
import GPyOpt
import logging
from typing import *
from .core import place_inside_mask
from .crossvalidation import split_list, bool_from_interval, corr_objective
from .defaults import ReconstructionConfig
from scipy import optimize, sparse
import matplotlib.pyplot as plt
from scipy.special import loggamma, erf, gammaln, digamma
from scipy.optimize import Bounds


def obj_logNbl1tv_grad(x, Acsc, b, alpha, beta, sa, sb, r, ixs):
    # print ("here")
    tv, grad_tv = tv_masked_dlasso_ddlasso(x, ixs, sb)
    b_hat = Acsc.dot(x)
    b_hat = b_hat + 1e-2 * b.mean()
    # print(b_hat.shape)
    obj = -sum_nb_ll(b, b_hat, r) + beta * tv + alpha * np.sum(dlasso(x, sa))
    dll = -grad_sum_nb_ll(b, b_hat, r, Acsc)

    grad = dll + beta * grad_tv + alpha * ddlasso(x, sa)

    # print (obj, grad)
    return obj, grad


def sum_nb_ll(b, mu, r):
    """
    compute negative binomial loglikelihood and gradient
    mu is b_pred = Acsc.dot(x)
    and b is y
    x
    """

    ## clip phi to make sure variance is small near the values that are small
    phi = r / mu
    phi_min = 1.0 / phi
    # phi_min = np.clip(phi_min, 0.1*np.mean(b)/r, None)
    lggamma_fun_ratio = gammaln(b + phi_min) - gammaln(phi_min) - gammaln(b + 1)
    log1muphi = np.log(1 + mu * phi)
    lgf1 = -phi_min * log1muphi
    lgf2 = b * (np.log(mu) - log1muphi + np.log(phi))
    return np.sum(lggamma_fun_ratio + lgf1 + lgf2)


def nb_loglik_(r, y, mu):
    psi = r / mu
    psi_min = 1.0 / psi
    lggamma_fun_ratio = loggamma(y + psi_min) - loggamma(psi_min) - loggamma(y + 1)
    log1mupsi = np.log(1 + mu * psi)
    lgf1 = -psi_min * log1mupsi
    lgf2 = y * (np.log(mu) - log1mupsi + np.log(psi))
    return -np.sum(lggamma_fun_ratio + lgf1 + lgf2)


def grad_sum_nb_ll(b, mu, r, Acsc):
    # mu = np.clip(mu)
    phi_min = mu / r
    phi_min = np.clip(phi_min, 0.1 * np.mean(b) / r, None)
    grad = -(1 / r) * (Acsc.T.dot(np.log(r + 1) + digamma(phi_min) - digamma(b + phi_min)))
    return grad


def psi_(x):
    sigma = 1.0 / np.sqrt(2.0)
    loglik = -np.log(sigma) - 0.5 * np.log(2 * np.pi) - 0.5 * (x / sigma) ** 2
    ### add if loglik arraz is less than some large negative number put it to zereo
    return np.exp(loglik)


def dlasso(x, s=0.005):
    return x * erf(x / s)


def ddlasso(x, s=0.005):
    return erf(x / s) + 2 * psi_(x / s) * x / s


def tv_masked_dlasso_ddlasso(x, ixs, s):
    ref_ixs_xy, ref_ixs_x, ref_ixs_y = ixs
    delta_x = x[ref_ixs_xy[:, 1]] - x[ref_ixs_xy[:, 0]]
    delta_y = x[ref_ixs_xy[:, 2]] - x[ref_ixs_xy[:, 0]]
    delta_x_border = x[ref_ixs_x[:, 1]] - x[ref_ixs_x[:, 0]]
    delta_y_border = x[ref_ixs_y[:, 1]] - x[ref_ixs_y[:, 0]]

    # New code
    # um = np.sum(dlasso(delta_x, s)) + np.sum(dlasso(delta_y, s)) # Add x and y
    # cum += np.sum(dlasso(delta_x_border, s)) + np.sum(dlasso(delta_y_border, s)) # Add x and y in border
    # Old code
    cum = np.sum(dlasso(delta_x, s))
    cum = cum + np.sum(dlasso(delta_y, s))
    cum = cum + np.sum(dlasso(delta_x_border, s)) + np.sum(dlasso(delta_y_border, s))
    grad = np.zeros(x.shape)
    grad[ref_ixs_xy[:, 0]] += ddlasso(-delta_x, s)
    grad[ref_ixs_xy[:, 1]] += ddlasso(delta_x, s)
    grad[ref_ixs_xy[:, 0]] += ddlasso(-delta_y, s)
    grad[ref_ixs_xy[:, 2]] += ddlasso(delta_y, s)
    grad[ref_ixs_x[:, 0]] += ddlasso(-delta_x_border, s)
    grad[ref_ixs_x[:, 1]] += ddlasso(delta_x_border, s)
    grad[ref_ixs_y[:, 0]] += ddlasso(-delta_y_border, s)
    grad[ref_ixs_y[:, 1]] += ddlasso(delta_y_border, s)
    return cum, grad


def make_tv_ixmask(mask):
    ref_ixs_xy = []
    ref_ixs_x = []
    ref_ixs_y = []
    rows, cols = mask.shape
    mask_num = -1 * np.ones(mask.shape, dtype=int)
    mask_num.flat[mask.flat[:].astype(bool)] = np.arange(np.sum(mask))
    for r in range(0, rows - 1):
        for c in range(0, cols - 1):
            ref, right, down = mask[r, c], mask[r, c + 1], mask[r + 1, c]
            if ref:
                if right and down:
                    ref_ixs_xy.append((mask_num[r, c], mask_num[r, c + 1], mask_num[r + 1, c]))
                elif right and not down:
                    ref_ixs_x.append((mask_num[r, c], mask_num[r, c + 1]))
                elif down and not right:
                    ref_ixs_y.append((mask_num[r, c], mask_num[r + 1, c]))
    ref_ixs_xy = np.array(ref_ixs_xy)
    ref_ixs_x = np.array(ref_ixs_x)
    ref_ixs_y = np.array(ref_ixs_y)

    return ref_ixs_xy, ref_ixs_x, ref_ixs_y


class ReconstructorFastScipyNB:
    def __init__(
        self,
        alpha: float = 1,
        beta: float = 0.01,
        config: ReconstructionConfig = ReconstructionConfig(),
        solver_kwargs: dict = {},
        ground_truth: np.ndarray = None,
        kfolds=3,
    ) -> None:
        # super().__init__(alpha, beta, config, solver_kwargs)
        self.solver_kwargs = solver_kwargs
        self.ground_truth = ground_truth
        self.w = 1
        self.sa = 0.005
        self.sb = 0.005
        self.cfg = config
        self.mask = self.cfg.mask_bw
        self.proj_N = self.cfg.proj_N
        self.A = self.cfg.A
        self._formulate(b=None)

        self.w = self.proj_N / (
            self.proj_N - 1
        )  # a weight to mantain the proportion regularized / RSS in crossvalidation

        self._formulate(b=None)
        self.formulated = True
        self.reformulated = False
        self.fit_at_least_once = False

        self.norm_factor = None

        self.norm_factor = 1
        self.r = None
        self.score_list = []
        if kfolds is None:
            self.kfolds = len(self.cfg.angles_names)
        self.kfolds = kfolds

        ixs = make_tv_ixmask(self.mask)

    def _score_reconstruction(self, x: np.ndarray, logged_grid=False) -> np.ndarray:
        if logged_grid:
            x = np.atleast_2d(2 ** x)  # optimization is performed in log2 space
        else:
            x = np.atleast_2d(x)

        fs = np.zeros((x.shape[0], 1))  # initialize ouput array
        SL = list(split_list(list(range(self.proj_N)), (self.proj_N - 1, 1)))  # splitted list
        for i in range(x.shape[0]):
            alpha, beta = x[i, :]

            logging.debug("I'm testing alpha= %.4f, beta=%.4f" % (alpha, beta))
            fs[i] = 0  # does not do anything
            self.change_par(alpha=alpha, beta=beta)
            self._A = np.array(self.A)
            lis = []
            for (train_list, test_list) in SL:
                trainset_bool = bool_from_interval(
                    train_list, self.cfg.boundaries, self.cfg.symmetry
                )
                testset_bool = bool_from_interval(
                    test_list, self.cfg.boundaries, self.cfg.symmetry
                )

                A_train, b_train = self.A[trainset_bool, :], np.copy(self.b)[trainset_bool]
                A_test, b_test = self.A[testset_bool, :], np.copy(self.b)[testset_bool]

                result = self._fit_train(b_train, A_train)
                b_test_pred = A_test.dot(result)

                lis.append(nb_loglik_(self.r, b_test, b_test_pred + 1e-2 * np.mean(b_test)))
            fs[i] = sum(lis) / len(SL)
            logging.debug(f"NBi={fs[i]}")
            self.score_list.append(fs[i])

        return fs

    def fit_predict(self, b: np.ndarray = None, warm_start: bool = True) -> np.ndarray:

        self.fit(b, A=self.A, warm_start=warm_start)
        return place_inside_mask(self.x, self.cfg.mask_bw)

    def fit(self, b: np.ndarray = None, A: np.ndarray = None, warm_start: bool = True) -> Any:

        if not self.formulated:
            self._formulate(b=b)
            self.solver_kwargs["warm_start"] = False
        else:
            self.solver_kwargs["warm_start"] = warm_start and self.fit_at_least_once
            self.change_par(b=b)

        x_guess = np.ones(self.A.shape[1]) * np.mean(self.b) / 10.0

        bounds = Bounds(np.zeros_like(x_guess), np.full_like(x_guess, np.sum(self.b) / 5.0))

        self.x = optimize.minimize(
            obj_logNbl1tv_grad,
            x0=x_guess,
            args=(self.A, self.b, self.alpha, self.beta, self.sa, self.sb, self.r, self.ixs),
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
        ).x
        self.fit_at_least_once = True
        return self

    def _fit_train(self, b: np.ndarray, A: np.ndarray, warm_start: bool = True) -> Any:
        """ Fit method
        Defines the model and fit it to the data.

        Args
        ----
        b: np.array, dtype float

        A: np.ndarray, dtype float

        mask: np.ndarray, dtype int | bool
            mask_gray > 0.1


        Return
        ------
        reconstructor: Recontructor
            The object after fit. To get the data access the attribute 'x' otherwise call fit_predict

        Note
        ----
        b, A and mask are not required if the proble has been previously formulated (e.g. if self.warmstart = True)
        """

        x_guess = np.ones(A.shape[1]) * np.mean(b) / 10.0

        bounds = Bounds(np.zeros_like(x_guess), np.full_like(x_guess, np.sum(b) / 5.0))

        x = optimize.minimize(
            obj_logNbl1tv_grad,
            x0=x_guess,
            args=(A, b, self.alpha, self.beta, self.sa, self.sb, self.r, self.ixs),
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
        ).x
        return x

    def optimize(
        self,
        b: np.ndarray,
        total_eval: int = 31,
        extra_evals: int = 10,
        logged_grid=True,
        initial_grid_n: int = 4,
        acquisition_par: float = 2,
        style: str = "gradient",
        gradient_iter: int = 5,
        max_time: int = None,
        domain: List[Dict] = None,
    ) -> None:

        logging.debug("Performing crossvalidation to determine alpha and beta")

        self._A = np.array(self.A)  # it makes a copy other than converting it to array

        if domain is None:
            raise (ValueError, "Please input domain")
        domain = [
            {"name": "alpha", "type": "continuous", "domain": domain[0]},
            {"name": "beta", "type": "continuous", "domain": domain[1]},
        ]

        logging.debug("To start, evaluate a small grid of %i points" % (initial_grid_n ** 2))
        domain2 = []
        internal_grid = []
        for n, i in enumerate(domain):
            delta = np.diff(i["domain"])[0]
            delta /= 100
            internal_grid.append(np.linspace(i["domain"][0], i["domain"][1], initial_grid_n))
            domain2.append(
                {
                    "name": ["alpha", "beta"][n],
                    "type": "continuous",
                    "domain": (i["domain"][0] + delta, i["domain"][1] - delta),
                }
            )

        self.change_par(b=b)
        Xs = np.vstack(
            [
                np.repeat(internal_grid[0], len(internal_grid[1])),
                np.tile(internal_grid[1], len(internal_grid[0])),
            ]
        ).T

        Ys = self._score_reconstruction(Xs, logged_grid=logged_grid)

        if extra_evals != 0:
            logging.debug(
                "Continue the optimum search by Bayessian Optimization (%i extra evaluations)"
                % (total_eval - initial_grid_n ** 2)
            )
            self.opt = GPyOpt.methods.BayesianOptimization(
                f=self._score_reconstruction,
                domain=domain2,  # box-constrains of the problem
                acquisition_type="LCB",  # LCB acquisition
                acquisition_par=acquisition_par,
                X=Xs,
                Y=Ys,
                maximize=False,
            )  # Minimize

            self.opt.run_optimization(max_iter=extra_evals, max_time=max_time)

        if logged_grid:
            alpha = 2 ** self.opt.x_opt[0]  # bring back to linear sclale
            beta = 2 ** self.opt.x_opt[1]
        else:
            alpha = self.opt.x_opt[0]
            beta = self.opt.x_opt[1]

        logging.debug("Optimum found alpha=%.4f , beta=%.4f " % (alpha, beta))
        self.alpha = self.w * alpha
        self.beta = self.w * beta
        logging.debug("Rescaled to alpha=%.4f , beta=%.4f " % (self.w * alpha, self.w * beta))

        self.change_par(b=b)

        if style == "gradient":
            self.change_par(b=b)
            Xs = np.array([[alpha, beta], [alpha + 0.25, beta], [alpha, beta + 0.25]])
            logging.debug("Gradient descent initialized for hyperparameter optimization")
            best_score = None
            for n in range(gradient_iter):
                # print(f'iteration: {n} \n')
                score_list = []
                for i in np.arange(Xs.shape[0]):
                    # logging.debug(f"The Xs is {Xs[i]}")
                    Ys = self._score_reconstruction(Xs[i], logged_grid=logged_grid)
                    if best_score == None:
                        best_score = (Ys, Xs[i])
                    elif Ys < best_score[0]:
                        best_score = (Ys, Xs[i])
                    score_list.append(Ys)
                score_list = np.array(score_list).squeeze()
                differences = [
                    score_list[0] - score_list[1],
                    score_list[0] - score_list[2],
                ]  # alpha difference then beta
                Xs += np.array(differences) / 100
                Xs = np.clip(Xs, 0, None)

            if extra_evals != 0:
                domain = [
                    {
                        "name": "alpha",
                        "type": "continuous",
                        "domain": (
                            np.nanmax(np.array([best_score[1][0] - 0.25, 0.0])),
                            best_score[1][0] + 0.25,
                        ),
                    },
                    {
                        "name": "beta",
                        "type": "continuous",
                        "domain": (
                            np.nanmax(np.array([best_score[1][1] - 0.25, 0.05])),
                            best_score[1][1] + 0.25,
                        ),
                    },
                ]

                logging.debug("Evaluate a small grid of %i points" % (initial_grid_n))
                # Evaluate a initial grid before starting the bayessian optimization procedure
                domain2 = []
                internal_grid = []
                for n, i in enumerate(domain):
                    delta = np.diff(i["domain"])[0]
                    delta /= 100
                    internal_grid.append(
                        np.linspace(i["domain"][0], i["domain"][1], initial_grid_n)
                    )
                    domain2.append(
                        {
                            "name": ["alpha", "beta"][n],
                            "type": "continuous",
                            "domain": (i["domain"][0] + delta, i["domain"][1] - delta),
                        }
                    )

                logging.debug(
                    "Continue the optimum search by Bayessian Optimization (%i extra evaluations)"
                    % (extra_evals)
                )
                fobj = lambda x: self._score_reconstruction(x, logged_grid=logged_grid)
                self.opt = GPyOpt.methods.BayesianOptimization(
                    f=fobj,  # function to optimize
                    domain=domain,  # box-constrains of the problem
                    acquisition_type="LCB",  # LCB acquisition
                    acquisition_par=acquisition_par,
                    # X=Xs_, Y=Ys_,
                    maximize=False,
                )  # Minimize

                self.opt.run_optimization(max_iter=extra_evals, max_time=max_time)
            if logged_grid:
                try:
                    alpha = 2 ** self.opt.x_opt[0]  # bring back to linear sclale
                    beta = 2 ** self.opt.x_opt[1]
                except:
                    alpha = 2 ** best_score[1][0]
                    beta = 2 ** best_score[1][1]
            else:
                try:
                    alpha = self.opt.x_opt[0]
                    beta = self.opt.x_opt[1]
                except:
                    alpha = best_score[1][0]
                    beta = best_score[1][1]

            logging.debug("Optimum found alpha=%.4f , beta=%.4f " % (alpha, beta))
            try:
                self.nb_score = self.opt.get_evaluations()[1].squeeze().min()
            except:
                self.nb_score = best_score[0]

            logging.debug(f"Lowest score found: {self.nb_score}")
            self.alpha = self.w * alpha
            self.beta = self.w * beta
            logging.debug(
                "Rescaled to alpha=%.4f , beta=%.4f " % (self.w * alpha, self.w * beta)
            )
            self.change_par(b=b)

    def change_par(
        self,
        alpha: float = None,
        beta: float = None,
        A: np.array = None,
        b: np.array = None,
        b_n: float = None,
    ) -> None:
        """ Change a parameter without reformulating the model.
        """
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        if A is not None:
            self.A = A
        if b is not None:
            self.b = b
        if b_n is not None:
            self.b_n = b_n

    def _formulate(self, b: np.ndarray = None) -> Any:

        if b is None:
            self.b = np.zeros(self.cfg.A.shape[0])
        # self.x = np.ones(self.cfg.A.shape[1]) * np.mean(self.b)/10.
        self.r = 0.0001
        self.A = sparse.csc_matrix(self.cfg.A)
        self.ixs = make_tv_ixmask(self.cfg.mask_bw)
