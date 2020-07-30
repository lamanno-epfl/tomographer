import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import *
from .core import build_Design_Matrix
from .gridsearch import zero_shifting, mixed_interpolator
from .utils import get_x, get_plate, mixed_interpolator2, colorize
from .tomographer import Tomographer
from skimage.color import gray2rgb
from matplotlib.colors import LogNorm, PowerNorm
from matplotlib import gridspec
from skimage.measure import find_contours


def plot_raw_data_sum(data: Dict[str, pd.DataFrame], spikes: Dict[str, pd.DataFrame]) -> None:
    """Plot a first diagnostic graf of the total molecules detected per projection, including spikeins
    """
    for i, (name_angle, df) in enumerate(data.items()):
        plt.subplot(151 + i)
        if i == 0:
            plt.ylabel("# Molecules")
        plt.plot(df.sum(0).values, c='b', lw=1.5)
        plt.plot(spikes[name_angle].sum(0).values, c='g', lw=1)
        plt.xlim(-1, df.shape[1] + 1)
        plt.title("%s\n%.1fM molecules\n%.1fMspikes" % (name_angle,
                                                        df.sum().sum() / 1000000.,
                                                        spikes[name_angle].sum().sum() / 1000000.), fontsize=10)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.tight_layout()


def plot_raw_spikes(spikes: Dict[str, pd.DataFrame]) -> None:
    for i, (name_angle, spikes_df) in enumerate(spikes.items()):
        plt.subplot(151 + i)
        if i == 0:
            plt.ylabel("# Molecules")
        plt.plot(spikes_df.sum(0).values, 'g+', ms=8)
        plt.xlim(-1, spikes_df.shape[1] + 1)
        plt.title("%s\n" % (name_angle))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.tight_layout()


def plot_gp_with_std(x: np.ndarray, y: np.ndarray, y_new: np.ndarray, y_std: np.ndarray, flag: Any) -> None:
    plt.plot(x, y, 'g+', ms=8)
    plt.plot(x, y_new, c='g', lw=3)
    plt.plot(x, y_new + y_std, c='gray', lw=1)
    plt.plot(x, y_new - y_std, c='gray', lw=1)
    plt.fill_between(x, y_new + y_std, y_new - y_std, color="gray", alpha=0.4)
    plt.plot(x, y_new - 1.96 * y_std, c='r', ls="--", lw=1)
    if type(flag) == list:
        for i in range(len(flag)):
            plt.plot(x[flag[i]], y[flag[i]], ["rx", "bv", "gs"][i % 3], fillstyle='none', ms=10)
    elif type(flag) == np.ndarray:
        plt.plot(x[flag], y[flag], "rx", ms=10)
    else:
        pass
    plt.xlim(-1, x.shape[0] + 1)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))


def plot_plate_adjustment(coordinates_dict: Dict[str, Tuple]) -> None:
    ''' Plot the results of plate normalization using gaussian processes

    Args
    ----
    coordinates_dict: Dict[Tuple]
        keys should be the angles
        the values tuple x, y, X_news, y_means, y_stds, f
        where each of them is a list of two elements containin the arrays corresponding to each of the plates
    '''
    # Plot how it looks after normalization
    gs = plt.GridSpec(2, len(coordinates_dict))
    for i, name_angle in enumerate(coordinates_dict.keys()):
        # unpack the dictionary
        x, y, X_news, y_means, y_stds, f = coordinates_dict[name_angle]
        # For each plate
        for j in range(len(x)):
            # Plot
            ax = plt.subplot(gs[0, i])
            # Plot the predicted gp before correction
            ax.scatter(x[j], y[j], marker=".", lw=0, s=35, c=f"C{j}")
            ax.plot(X_news[j], y_means[j], 'k', lw=1, zorder=9, label="predicted mean")
            ax.fill_between(X_news[j].flat[:], y_means[j] + y_stds[j], y_means[j] - y_stds[j], alpha=0.2, color=f"C{j}")
            ax.set_title("%s" % name_angle)
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            plt.ylim(0, max(np.max(y[k]) for k in range(len(x))) * 1.05)
            plt.xlim(-4, max(x[-1]) + 4)
            # Plot the predicted gp after correction
            ax = plt.subplot(gs[1, i])
            ax.scatter(x[j], y[j] * f[j], marker=".", lw=0, s=35, c=f"C{j}")
            ax.plot(X_news[j].flat[:], y_means[j] * f[j], 'k', lw=1, zorder=9, label="predicted mean")
            ax.fill_between(X_news[j].flat[:], (y_means[j] + y_stds[j]) * f[j], (y_means[j] - y_stds[j]) * f[j], alpha=0.2, color=f"C{j}")
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            plt.ylim(0, max(np.max(y[k]) for k in range(len(x))) * 1.05)
            plt.xlim(-4, max(x[-1]) + 4)
    plt.tight_layout()


def plot_spikes_adjustment(coordinates_dict: Dict[str, Tuple]) -> None:
    ''' Plot the results of spikes normalization using gaussian processes

    Args
    ----
    coordinates_dict: Dict[Tuple]
        keys should be the angles
        the values tuple x, y, X_news, y_means, y_stds, f
        where each of them is a list of two elements containin the arrays corresponding to each of the plates
    '''
    # Plot how it looks after normalization
    gs = plt.GridSpec(2, len(coordinates_dict))
    for i, name_angle in enumerate(coordinates_dict.keys()):
        # unpack the dictionary
        x, y, X_news, y_pred, y_stds, adj = coordinates_dict[name_angle]
        # For each plate
        for j in range(len(x)):
            # Plot
            ax = plt.subplot(gs[0, i])
            # Plot the predicted gp before correction
            ax.scatter(x[j], y[j], marker=".", lw=0, s=35, c=f"C{j}")
            ax.plot(X_news[j], y_pred[j], 'k', lw=1, zorder=9, label="predicted mean")
            ax.fill_between(X_news[j].flat[:], y_pred[j] + y_stds[j], y_pred[j] - y_stds[j], alpha=0.2, color=f"C{j}")
            ax.set_title("%s" % name_angle)
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            plt.ylim(0, max(np.max(y[k]) for k in range(len(x))) * 1.05)
            plt.xlim(-4, max(x[-1]) + 4)
            # Plot the predicted gp after correction
            ax = plt.subplot(gs[1, i])
            ax.scatter(x[j], y[j] * adj[j], marker=".", lw=0, s=35, c=f"C{j}")
            ax.plot(X_news[j].flat[:], y_pred[j] * adj[j], 'k', lw=1, zorder=9, label="predicted mean")
            ax.fill_between(X_news[j].flat[:], y_pred[j] * adj[j] + y_stds[j], y_pred[j] * adj[j] - y_stds[j], alpha=0.2, color=f"C{j}")
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            plt.ylim(0, max(np.max(y[k]) for k in range(len(x))) * 1.05)
            plt.xlim(-4, max(x[-1]) + 4)
    plt.tight_layout()


def plot_opt_results(opt_results: Dict[str, Any], data_norm: Dict[str, pd.DataFrame], ref_im: np.ndarray,
                     angles_dict: Dict[str, float], first_points: Dict[str, int], widths_dict: Dict[str, Any], mask: np.ndarray) -> None:
    gs = plt.GridSpec(2 * len(ref_im), 6)

    for i, gene in enumerate(ref_im.keys()):
        # i -> gene
        plt.subplot(gs[2 * i:2 * (i + 1), 0])
        # plot image
        plt.imshow(ref_im[gene] * mask)
        for j, angle in enumerate(opt_results[gene].keys()):
            # Plot gene at the optimized angle
            plt.subplot(gs[2 * i + 1, j + 1])
            ix = np.argmin(opt_results[gene][angle].ix[:, "objective"])
            results = opt_results[gene][angle].ix[ix, :]
            _D, _projs_len = build_Design_Matrix(angles=np.array([results["angle"]]),
                                                 widths=[results["width"]],
                                                 image=1. * (mask >= 0.2))
            theorProj = _D.dot((ref_im[gene] * mask).flat[:])
            plt.title("angle: %.1f shift: %d, wid: %.2f\n%s" % (np.rad2deg(results["angle"]), results["shift"], results["width"], angle))
            plt.plot(results["scaling"] * theorProj, "k", zorder=1000, lw=2)
            plt.plot(zero_shifting(mixed_interpolator(get_x(data_norm[angle]), data_norm[angle].ix[gene].values), int(results["shift"])), 'r.--', alpha=0.6, ms=10)
            plt.ticklabel_format(style='sci', axis='y', scilimits=(1, 2))
            plt.gca().yaxis.major.formatter._useMathText = True
            
            # Plot gene at the default angle
            plt.subplot(gs[2 * i, j + 1])
            _D, _projs_len = build_Design_Matrix(angles=np.array([angles_dict[angle]]),
                                                 widths=[widths_dict[angle]],
                                                 image=1. * (mask >= 0.2))
            theorProj = _D.dot((ref_im[gene] * mask).flat[:])
            plt.title("angle: %.1f shift: %d, wid: %.2f\n%s" % (np.rad2deg(angles_dict[angle]), -first_points[angle], widths_dict[angle], angle))
            plt.plot(results["scaling"] * theorProj, "k", zorder=1000, lw=2)
            plt.plot(zero_shifting(mixed_interpolator(get_x(data_norm[angle]), data_norm[angle].ix[gene].values), -first_points[angle]), 'r.--', alpha=0.6, ms=10)
            plt.ticklabel_format(style='sci', axis='y', scilimits=(1, 2))
            plt.gca().yaxis.major.formatter._useMathText = True
        plt.tight_layout()


def plot_projection_check(gene: str, data_norm: Dict[str, pd.DataFrame], angles: np.ndarray, widths: np.ndarray, first_points_dict: Dict[str, int], mask: np.ndarray) -> None:
    D, projs_len = build_Design_Matrix(angles, widths, mask)
    boundaries = np.r_[0, np.cumsum(projs_len)]

    gs = plt.GridSpec(2, 6)
    imgs = []
    for i, (name_angle, df_angle) in enumerate(data_norm.items()):
        projected_total = D.dot(mask.flat[:])[boundaries[i]:boundaries[i + 1]]
        x = get_x(data_norm[name_angle]).astype(int)
        y = data_norm[name_angle].ix[gene]
        
        x, y, not_provided = mixed_interpolator2(x, y)
        c = np.array(["g"] * len(x))
        c[~not_provided] = np.array(["g", "r", "b", "y"])[get_plate(data_norm[name_angle])]

        # Set the plot
        ax = plt.subplot(gs[0, i + 1])
        # Plot points
        ax.scatter(x, y, c=c, marker=".", lw=0, s=50)
        ax.plot(x, y, alpha=0.2, c="gray")
        scaling = np.percentile(y, 95) / projected_total.max()
        # Plot expected projection
        ax.plot(np.arange(projected_total.shape[0]) + first_points_dict[name_angle], projected_total * scaling, c="k", lw=1.2, alpha=0.5, zorder=1000)
        # Fix graphics
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.set_title("%s" % name_angle)
        ax.set_xlim(-5, get_x(df_angle)[-1] + 5)
        ax.set_ylim(-0.2,)
        
        # Set the plot
        ax = plt.subplot(gs[1, i + 1])

        # Identify the part of the Design matrix containing the strips relative to one angle
        s, e = boundaries[i], boundaries[i + 1]
        part_D = D[s:e]
        # identify the detected level of expression to multiply to the corresponding strips
        detected_at_slice = np.zeros(part_D.shape[0])
        ix_detected = np.in1d(x - int(first_points_dict[name_angle]), np.arange(e - s))  # take care of the case where x has gaps
        ix_reference = (x - int(first_points_dict[name_angle]))[ix_detected]
        detected_at_slice[ix_reference] = y[ix_detected]
        # Perform the multiplication desigm_matrix@angle * level_expression@angle, then sum pixelwise and reshape as an image
        img0 = (0.5 * part_D * detected_at_slice[:, None]).sum(0).reshape(mask.shape)
        imgs.append(img0)
        ax.imshow(10 * img0 / img0.sum(), cmap="viridis", norm=PowerNorm(0.5))  # vmin=np.percentile(img0, 2), vmax=np.percentile(img0, 98), cmap="gray")
    
    # Plot the product of all the images in lognormalized colormap
    ax = plt.subplot(gs[0:2, 0])
    res_img = np.ones_like(imgs[0])
    for img in imgs:
        res_img *= img + 1e-5
    ax.set_title(gene)
    ax.imshow(10 * res_img / res_img.sum(), cmap="viridis", norm=PowerNorm(0.25))
    plt.tight_layout()


def show_reconstruction_raw(result: np.ndarray, mask: np.ndarray, ax: Any=None) -> None:
    cntrs = find_contours(mask, 0.1)[0]
    if ax is None:
        ax = plt.subplot(111)
    ax.plot(cntrs[:, 1], cntrs[:, 0], lw=1, c='k', ls='--')
    ax.imshow(result * mask, cmap="gray_r", interpolation="none")
    ax.tick_params(axis='both', labelleft='off', labelbottom="off")
    ax.axis("off")


def mixing_with_img(result: np.ndarray, reference_mask: np.ndarray, mask: np.ndarray, mixing_factor: float=1.5, hue: float=0.28, saturation: float=1) -> np.ndarray:
    cum_image = 0.6 * reference_mask.copy()[:, :, :3]
    image = result * mask / np.max(result * mask)
    image = gray2rgb(image * reference_mask[:, :, 0])
    image = colorize(image, hue=hue, saturation=saturation)  # green
    cum_image = np.maximum(mixing_factor * image, cum_image)
    return np.clip(cum_image, 0, 1)


def show_reconstruction(result: np.ndarray, reference_mask: np.ndarray, mask: np.ndarray, ax: Any = None, mixing_factor: float=1.5) -> None:
    if ax is None:
        ax = plt.subplot(111)
    cum_image = mixing_with_img(result=result, reference_mask=reference_mask, mask=mask, mixing_factor=mixing_factor)
    ax.imshow(cum_image)
    ax.axis("off")
    plt.tight_layout(pad=0, h_pad=0)


def plot_reconstruction_check(gene: str, data_norm: Dict[str, pd.DataFrame], b: np.ndarray, prediction_projection: np.ndarray,
                              total_projection: np.ndarray, first_points: Dict[str, int], boundaries: np.ndarray, redlist: Iterable=()) -> None:
    # projection_prediction = A.dot(result)
    # boundaries = np.r_[0, np.cumsum(projs_len)]
    gs = plt.GridSpec(1, len(data_norm))
    for i, name_angle in enumerate(first_points.keys()):
        n = boundaries[i + 1] - boundaries[i]
        
        # Find the plate colors in a little bit hacky way
        xi = get_x(data_norm[name_angle])
        plate_id = get_plate(data_norm[name_angle])
        col_bound = xi[np.where(np.diff(plate_id))[0][0]]
        c_vals = np.ones(shape=(n,)).astype(int)
        c_vals[int(col_bound - first_points[name_angle] + 1):] += 1
        c = np.array(["k", "r", "b", "y"])[c_vals]

        # Plot data trace
        x = np.arange(n)
        y = (b / b.max())[boundaries[i]:boundaries[i + 1]]
        ax = plt.subplot(gs[i])
        ax.plot(x, y, color="gray", label='real data')
        ax.scatter(x, y, c=c, lw=0.1, s=10)

        # Plot Reconstructed trace
        y_reconstructed = prediction_projection[boundaries[i]:boundaries[i + 1]]
        if i in redlist:
            ax.plot(x, y_reconstructed, 'r', lw=2.5, label='reconstructed')
        else:
            ax.plot(x, y_reconstructed, 'g', lw=1.7, label='reconstructed')
        
        # Plot the backgroud sum of pixesl for reference
        y = total_projection[boundaries[i]:boundaries[i + 1]] / total_projection.max()
        ax.fill_between(x, y, y2=0, color="gray", lw=1.5, zorder=-1, alpha=0.1)
        
        # Graphical tweeks of the plot
        plt.ylim(0, 1.02)
        plt.xlim(n / 2. - max(np.diff(boundaries)) / 1.9, n / 2. + max(np.diff(boundaries)) / 1.9)
        if i == 0:
            plt.ylabel('Normalized reads count')
            ax.tick_params(axis='y', labelleft='on', right="off")
        else:
            ax.tick_params(axis='y', labelleft='off', right="off")
        plt.title('%s' % name_angle, fontsize=12)
    plt.tight_layout(pad=0.0, w_pad=0.0)


def plot_projections_recontruction(tg: Tomographer, ss: gridspec.SubplotSpec=None) -> None:
    if ss is None:
        gs = gridspec.GridSpec(1, tg.cfg.proj_N)
    else:
        gs = gridspec.GridSpecFromSubplotSpec(1, tg.cfg.proj_N, subplot_spec=ss)
    for i, name_angle in enumerate(tg.cfg.angles_names):
        n = tg.cfg.boundaries[i + 1] - tg.cfg.boundaries[i]
        
        # Find the plate colors in a little bit hacky way
        xi = tg.data.xs[i]

        # Plot data trace
        b = np.array(tg.reconstructor.b.value)
        x = np.arange(n)
        y = b[tg.cfg.boundaries[i]:tg.cfg.boundaries[i + 1]]
        ax = plt.subplot(gs[i])
        ax.plot(x, y, color="gray", label='real data', alpha=0.5)
        ax.scatter(x, y, c="g", lw=0.1, s=20, zorder=10)

        # Plot Reconstructed trace
        prediction_projection = tg.cfg.A.dot(np.array(tg.reconstructor.x.value).flat[:])
        y_reconstructed = prediction_projection[tg.cfg.boundaries[i]:tg.cfg.boundaries[i + 1]]
        ax.plot(x, y_reconstructed, color="k", lw=2.5, label='reconstructed', zorder=1000)
        # Plot the backgroud sum of pixesl for reference
        
        total_projection = tg.cfg.A.dot(np.ones(tg.reconstructor.x.value.shape[0]))
        y_area = total_projection[tg.cfg.boundaries[i]:tg.cfg.boundaries[i + 1]] / total_projection.max()
        ax.fill_between(x, y_area, y2=0, color="gray", lw=0, zorder=-1, alpha=0.1)
        
        # Graphical tweeks of the plot
        plt.ylim(0, np.maximum(np.max(y), np.max(y_reconstructed)) + 0.05)
        plt.xlim(n / 2. - max(np.diff(tg.cfg.boundaries)) / 1.9, n / 2. + max(np.diff(tg.cfg.boundaries)) / 1.9)
        if i == 0:
            plt.ylabel('Normalized reads count')
            ax.tick_params(axis='y', labelleft='on', right="off")
        else:
            ax.tick_params(axis='y', labelleft='off', right="off")
        plt.title('%s' % name_angle, fontsize=18)
    plt.tight_layout(pad=0.0, w_pad=0.0)
