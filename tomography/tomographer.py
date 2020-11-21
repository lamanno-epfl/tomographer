from typing import *
import logging
import warnings
import numpy as np
from .defaults import load_config, ReconstructionConfig, connect_to_data, DataConnection
from .core import build_Design_Matrix, prepare_design_masked, prepare_design_symmetry, prepare_design_symmetry_masked, prepare_observations, prepare_observations_symmetry
from .optimization import ReconstructorFastScipyNB


class Tomographer:
    """Main object to work with tomographic reconstruction problems

    The analysis steps are:
    
    #. First load the defaults

        * tg = Tomographer()
        * tg.load_cfg("config_file.hdf5")  # or set this manually
        * tg.prepare_design()

    #. Then load the data

        * tg.connect_data("data.hdf5")

    #. Finally, reconstruct one gene

        * tg.reconstruct(gene="Lmx1a")
    """

    def __init__(self) -> None:
        self.cfg = ReconstructionConfig()  # type: ReconstructionConfig
        self.data = None  # type: DataConnection
        self.reconstructor = None  # type: Reconstructor
        self.ready_flag = False
    
    def load_cfg(self, cfg: Union[str, ReconstructionConfig]) -> None:
        """Load defaults from file
        Args
        ----
        defaults: filename (str) or ReconstructionDefaults (object)
        """
        if type(cfg) == str:
            self.cfg = load_config(cfg)
        else:
            assert type(cfg) == ReconstructionConfig
            self.cfg = cfg

    def connect_data(self, datafile: str) -> None:
        """Load hdf5 dataset
        """
        self.data = connect_to_data(datafile)
        self.data.bind(self.cfg)

    def prepare_design(self, symmetry: bool=None, masked_formulation: bool=None, kfolds: int = 5) -> None:
        if symmetry is not None:
            self.cfg.symmetry = symmetry
        if masked_formulation is not None:
            self.cfg.masked_formulation = masked_formulation

        D, proj_len = build_Design_Matrix(self.cfg.angles_values,
                                          self.cfg.widths,
                                          self.cfg.mask_g,
                                          self.cfg.mask_thrs,
                                          notation_inverse=True,
                                          return_projlen=True)
        self.cfg.proj_len = proj_len
        
        if self.cfg.symmetry:
            self.prepare = prepare_observations_symmetry
            if self.cfg.masked_formulation:
                self.cfg.A = prepare_design_symmetry_masked(D, self.cfg.mask_bw)
                self.reconstructor = ReconstructorFastScipyNB(config=self.cfg)
            else:
                self.cfg.A = prepare_design_symmetry(D)
                self.reconstructor = Reconstructor(config=self.cfg)
        else:
            self.prepare = prepare_observations
            if self.cfg.masked_formulation:
                self.cfg.A = prepare_design_masked(D, self.cfg.mask_bw)
                self.reconstructor = ReconstructorFastScipyNB(config=self.cfg, kfolds=kfolds)
            else:
                self.cfg.A = D
                self.reconstructor = Reconstructor(config=self.cfg)

        self.ready_flag = True

    def reconstruct(self, gene: str, alpha_beta: Union[str, Tuple[float, float]]="auto",
                    warm_start: bool=False,
                    crossval_kwargs: Dict[str, Any]={}, nb_r = 0.3, kfolds=3) -> np.ndarray:
        """Perform reconstruction of a gene

        Args
        ----
        gene: str
            gene name as in datafile

        alpha_beta: str or tuple
            options are:
                "auto": it will try to use the alpha and beta parameters present in the data file, if none is provided it will perform crossvalidation
                "crossvalidation": it will force crossvalidation
                (alpha, beta): passing a tuple will perform the reconstruction using this parameters

        warm_start:
            whether to use the previous reconstruction result as warm start.
            NOTE what the previous reconstruction is might change depending the alpha_beta parameter
        
        Return
        ------
        reconstructed: np.ndarray (2d)
            the reconstructed signal already reshaped


        NOTE you can also access the recosntruction parameters using the attribute `.reconstructor`
        """
        if not self.ready_flag:
            warnings.warn("prepare_design should be called before attempting reconstruction otherwise it is called automatically with default arguments!", UserWarning)
            self.prepare_design(kfolds=kfolds)

        logging.info("Preparing Reconstruction")
        b = self.prepare(projections=self.data[gene], xs=self.data.xs,
                         first_points=self.cfg.first_points, projs_len=self.cfg.proj_len,
                         interpolation="linear")

        # Normalize b taking some precaoution against high outlier and against division by 0
        # More naive form: b / b.max()
        self.unnormalized_b = np.copy(b)

        if type(self.reconstructor) is ReconstructorFastScipyNB:
            self.norm_factor = 1
            self.reconstructor.r = nb_r
        else:
            self.norm_factor = np.clip(np.percentile(b, 98), np.max(b) / 3., np.inf)
            b = self.unnormalized_b / self.norm_factor
        
        self.reconstructor.norm_factor = self.norm_factor

        if alpha_beta == "auto":
            try:
                _alpha, _beta = self.data.alphas_betas[gene]
                logging.info("alpha and beta found in the data file")
                self.reconstructor.change_par(alpha=_alpha, beta=_beta)
            except KeyError:
                logging.debug("alpha and beta not found in the data file, performing crossvalidation")
                self.reconstructor.optimize(b, **crossval_kwargs)
        elif alpha_beta == "crossvalidation":
            self.reconstructor.optimize(b, **crossval_kwargs)
        elif isinstance(alpha_beta, tuple) or isinstance(alpha_beta, list) or isinstance(alpha_beta, np.array):
            _alpha, _beta = alpha_beta
            self.reconstructor.change_par(alpha=_alpha, beta=_beta)
        logging.info("Reconstructing")
        result = self.reconstructor.fit_predict(b, warm_start=warm_start)
        logging.info("Finished Reconstructing %s" % gene)
        return result


class TomographerDebug(Tomographer):
    def __init__(self) -> None:
        super().__init__()

    def prepare_design(self, symmetry: bool=None, masked_formulation: bool=None) -> None:
        if symmetry is not None:
            self.cfg.symmetry = symmetry
        if masked_formulation is not None:
            self.cfg.masked_formulation = masked_formulation

        D, proj_len = build_Design_Matrix(self.cfg.angles_values,
                                          self.cfg.widths,
                                          self.cfg.mask_g,
                                          self.cfg.mask_thrs,
                                          notation_inverse=True,
                                          return_projlen=True)
        self.cfg.proj_len = proj_len
        
        if self.cfg.symmetry:
            self.prepare = prepare_observations_symmetry
            if self.cfg.masked_formulation:
                self.cfg.A = prepare_design_symmetry_masked(D, self.cfg.mask_bw)
                self.reconstructor = ReconstructorFastTest(config=self.cfg)
            else:
                self.cfg.A = prepare_design_symmetry(D)
                self.reconstructor = Reconstructor(config=self.cfg)
        else:
            self.prepare = prepare_observations
            if self.cfg.masked_formulation:
                self.cfg.A = prepare_design_masked(D, self.cfg.mask_bw)
                self.reconstructor = ReconstructorFastTest(config=self.cfg)
            else:
                self.cfg.A = D
                self.reconstructor = Reconstructor(config=self.cfg)

        self.ready_flag = True

    def reconstruct(self, gene: str, alpha_beta: Union[str, Tuple[float, float]]="auto",
                    warm_start: bool=False,
                    crossval_kwargs: Dict[str, Any]={}) -> np.ndarray:
        """Perform reconstruction of a gene

        Args
        ----
        gene: str
            gene name as in datafile

        alpha_beta: str or tuple
            options are:
                "auto": it will try to use the alpha and beta parameters present in the data file, if none is provided it will perform crossvalidation
                "crossvalidation": it will force crossvalidation
                (alpha, beta): passing a tuple will perform the reconstruction using this parameters

        warm_start:
            whether to use the previous reconstruction result as warm start.
            NOTE what the previous reconstruction is might change depending the alpha_beta parameter
        
        Return
        ------
        reconstructed: np.ndarray (2d)
            the reconstructed signal already reshaped


        NOTE you can also access the recosntruction parameters using the attribute `.reconstructor`
        """
        if not self.ready_flag:
            warnings.warn("prepare_design should be called before attempting reconstruction otherwise it is called automatically with default arguments!", UserWarning)
            self.prepare_design()

        logging.info("Preparing Reconstruction")
        b = self.prepare(projections=self.data[gene], xs=self.data.xs,
                         first_points=self.cfg.first_points, projs_len=self.cfg.proj_len,
                         interpolation="linear")

        # Normalize b taking some precaoution against high outlier and against division by 0
        # More naive form: b / b.max()
        self.unnormalized_b = np.copy(b)
        self.norm_factor = np.maximum(np.percentile(b, 98), np.max(b) / 3.)
        # hopefully this is the count data as i comment ourt the self.norm_factor lines 
        b = self.unnormalized_b / self.norm_factor
        self.reconstructor.norm_factor = self.norm_factor

        if alpha_beta == "auto":
            try:
                _alpha, _beta = self.data.alphas_betas[gene]
                logging.info("alpha and beta found in the data file")
                self.reconstructor.change_par(alpha=_alpha, beta=_beta)
            except KeyError:
                logging.debug("alpha and beta not found in the data file, performing crossvalidation")
                self.reconstructor.optimize(b, **crossval_kwargs)
        elif alpha_beta == "crossvalidation":
            self.reconstructor.optimize(b, **crossval_kwargs)
        elif isinstance(alpha_beta, tuple) or isinstance(alpha_beta, list) or isinstance(alpha_beta, np.array):
            _alpha, _beta = alpha_beta
            self.reconstructor.change_par(alpha=_alpha, beta=_beta)
        logging.info("Reconstructing")
        result = self.reconstructor.fit_predict(b, warm_start=warm_start)
        logging.info("Finished Reconstructing %s" % gene)
        return result




class TomographerManualDesignMatrix(Tomographer):
    def __init__(self, A: np.ndarray, projection_lengths: list) -> None:
        super().__init__()
        self.A_manual = A
        self.projection_lengths = projection_lengths
        
        
       # self.reconstructor = ReconstructorFastScipyNB()


    def prepare_design(self, symmetry: bool=None, masked_formulation: bool=None) -> None:
        if symmetry is not None:
            self.cfg.symmetry = symmetry
        if masked_formulation is not None:
            self.cfg.masked_formulation = masked_formulation

        self.cfg.A = self.A_manual
        D = self.A_manual
        self.cfg.proj_len =  self.projection_lengths #proj_len
        
        if self.cfg.symmetry:
            self.prepare = prepare_observations_symmetry
            if self.cfg.masked_formulation:
                #self.cfg.A = prepare_design_symmetry_masked(D, self.cfg.mask_bw)
                self.reconstructor = ReconstructorFastScipyNB(config=self.cfg)
            else:
                #self.cfg.A = prepare_design_symmetry(D)
                self.reconstructor = ReconstructorFastScipyNB(config=self.cfg)
        else:
            self.prepare = prepare_observations
            if self.cfg.masked_formulation:
                #self.cfg.A = prepare_design_masked(D, self.cfg.mask_bw)
                self.reconstructor = ReconstructorFastScipyNB(config=self.cfg)
            else:
                #self.cfg.A = D
                self.reconstructor = ReconstructorFastScipyNB(config=self.cfg)

        self.cfg.A = self.A_manual
        self.ready_flag = True


    def reconstruct(self, b: np.ndarray, alpha_beta: Union[str, Tuple[float, float]]="auto",
                    warm_start: bool=False,
                    crossval_kwargs: Dict[str, Any]={}) -> np.ndarray:
        """Perform reconstruction of a gene

        Args
        ----
        gene: str
            gene name as in datafile

        alpha_beta: str or tuple
            options are:
                "auto": it will try to use the alpha and beta parameters present in the data file, if none is provided it will perform crossvalidation
                "crossvalidation": it will force crossvalidation
                (alpha, beta): passing a tuple will perform the reconstruction using this parameters

        warm_start:
            whether to use the previous reconstruction result as warm start.
            NOTE what the previous reconstruction is might change depending the alpha_beta parameter
        
        Return
        ------
        reconstructed: np.ndarray (2d)
            the reconstructed signal already reshaped


        NOTE you can also access the recosntruction parameters using the attribute `.reconstructor`
        """
        


        #b = self.observations[gene] # select observation vector

        # Normalize b taking some precaoution against high outlier and against division by 0
        # More naive form: b / b.max()
        self.unnormalized_b = np.copy(b)
        self.norm_factor = np.maximum(np.percentile(b, 98), np.max(b) / 3.)
        # hopefully this is the count data as i comment ourt the self.norm_factor lines 
        b = self.unnormalized_b / self.norm_factor

        #self.reconstructor.norm_factor = self.norm_factor

        if alpha_beta == "auto":
            try:
                _alpha, _beta = self.data.alphas_betas[gene]
                logging.info("alpha and beta found in the data file")
                self.reconstructor.change_par(alpha=_alpha, beta=_beta)
            except KeyError:
                logging.debug("alpha and beta not found in the data file, performing crossvalidation")
                self.reconstructor.optimize(b, **crossval_kwargs)
        elif alpha_beta == "crossvalidation":
            self.reconstructor.optimize(b, **crossval_kwargs)
        elif isinstance(alpha_beta, tuple) or isinstance(alpha_beta, list) or isinstance(alpha_beta, np.array):
            _alpha, _beta = alpha_beta
            self.reconstructor.change_par(alpha=_alpha, beta=_beta)
        logging.info("Reconstructing")
        result = self.reconstructor.fit_predict(b, warm_start=warm_start)
        logging.info("Finished Reconstructing %")
        return result
