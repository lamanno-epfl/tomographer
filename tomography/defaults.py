from typing import *
import h5py
import logging
import os.path
import numpy as np
import os.path


def list_str_to_safe_array(attribute: List[str]) -> np.ndarray:
    return np.array([x.encode('ascii', 'ignore') for x in attribute])


# name attribute, function to call before saving, fucntion to call at load
CONFIG_FILE_SPEC = [("angles_names", list_str_to_safe_array, lambda x: x.astype("str")),
                    ("mask_g", lambda x: x, lambda x: x),
                    ("mask_thrs", lambda x: np.array([x]), lambda x: x[0]),
                    ("reference_mask", lambda x: x, lambda x: x),
                    ("symmetry", lambda x: np.array([x]), lambda x: bool(x[0])),
                    ("masked_formulation", lambda x: np.array([x]), lambda x: bool(x[0])),
                    ("angles_values", lambda x: np.array(x), lambda x: x),
                    ("first_points", lambda x: np.array(x), lambda x: list(x)),
                    ("widths", lambda x: np.array(x), lambda x: list(x)),
                    ("proj_len", lambda x: np.array(x), lambda x: list(x))]


class ReconstructionConfig:
    """Class to construct config files for Tomographer

    Args
    ----
    angle_names: List[str] 
        list containing names of angles
    mask: 2D array
        contains mask of tissue
    mask_thrs: Float, default 0.2
        threshold for reference image used to create mask
    reference_mask: 2D array
        reference image
    symmetry: Boolean, default False
        value indicating whether design matrix should be built assuming symmetry in tissue
    masked_formulation: Boolean, default True
        value indicating whether design matrix should be built using the mask
    angles_values: Array
        array containing angles in radians
    first_points: List[int]
        list specifying the starting point to begin considering values within projection for each angle
    widths: List[float]
        list indicating the estimated widths of the secondary slices in each angle

    Attributes
    -----
    angles_names
    mask_g
    mask_thrs
    reference_mask
    symmetry
    masked_formulation
    angles_values
    first_points
    widths
    proj_len
    A

    Methods
    -------
    to_file
    """

    def __init__(self, angles_names: List[str]=None,
                 mask: np.ndarray=None, mask_thrs: float=0.2, reference_mask: np.ndarray=None,
                 symmetry: bool=False, masked_formulation: bool=True,
                 angles_values: np.ndarray=None, first_points: List[int]=None, widths: List[float]=None) -> None:

        # This are the primary defaults, should be considered as constants, they are defined for each experiments
        self.angles_names = angles_names
        self.mask_g = mask
        self.mask_thrs = mask_thrs
        self.reference_mask = reference_mask
        self.symmetry = symmetry
        self.masked_formulation = masked_formulation

        # This are secondary defaults, since  different values can yield a better or worse reconstruction
        self.angles_values = angles_values
        self.first_points = first_points
        self.widths = widths

        # Empty at initialization, they are computed on the basis of the primary and secondary defaults
        self.proj_len = None  # type: List[int]
        self.A = None  # type: np.ndarray

    @property
    def boundaries(self) -> np.ndarray:
        return np.r_[0, np.cumsum(self.proj_len)]
    
    @boundaries.setter
    def boundaries(self, value: np.ndarray) -> None:
        self.proj_len = np.diff(value)

    @property
    def proj_N(self) -> int:
        return len(self.proj_len)

    @proj_N.setter
    def proj_N(self, value: int) -> None:
        raise ValueError("proj_N cannot be set it is inferred from proj_len")

    @property
    def mask_bw(self) -> np.ndarray:
        return (self.mask_g > self.mask_thrs).astype(float)

    @mask_bw.setter
    def mask_bw(self, value: np.ndarray) -> None:
        raise ValueError("mask_bw cannot be set and it is obtained by thresholding mask_g. Set mask_g and mask_thrs instead")

    def to_file(self, filename: str) -> None:
        if os.path.isfile(filename):
            raise IOError("File exist, cannot overwrite")
        f = h5py.File(filename, "w")
        for attribute_name, fun_out, _ in CONFIG_FILE_SPEC:
            try:
                attribute = getattr(self, attribute_name)
                if attribute is None:
                    logging.warn("%s entry was None, it will not be seved to file" % attribute_name)
                else:
                    f.create_dataset(attribute_name, data=fun_out(attribute))
            except AttributeError:
                logging.warning("Entry %s does not exist, it will be skipped." % attribute_name)
            

def load_config(filename: str) -> ReconstructionConfig:
    if not os.path.isfile(filename):
        raise IOError("%s is not a file" % filename)
    cfg = ReconstructionConfig()
    f = h5py.File(filename, "r")
    logging.debug("Loding reconstruction defaults")
    for attribute_name, _, fun_in in CONFIG_FILE_SPEC:
        try:
            loaded = f[attribute_name][:]
            setattr(cfg, attribute_name, fun_in(loaded))
        except KeyError:
            logging.warn("Dataset %s was not found in file. The default %s will be loaded" % (attribute_name, getattr(cfg, attribute_name)))
    f.close()
    return cfg


class AlphaBetaFetcher:
    def __init__(self, file_handle: h5py.File) -> None:
        self.f = file_handle
        try:
            self.alphas_betas = self.f["alphas_betas"][:]
        except KeyError:
            self.alphas_betas = None
        try:
            self.genes_order = list(self.f["genes_order"][:].astype(str))
        except KeyError:
            if self.alphas_betas is None:
                logging.warn('gene_order dataset entry was not find in file.\n It might results in errors \nFilling with sorted(list(self.f["genes"].keys()))')
                self.genes_order = sorted(list(self.f["genes"].keys()))
            else:
                logging.error("specifying alphas_betas requires a genes_order attribute")
                raise IOError

    def __getitem__(self, gene: str) -> Tuple[float, float]:
        if self.alphas_betas is None:
            raise KeyError
        values = self.alphas_betas[self.genes_order.index(gene), :]
        if np.any(np.isnan(values)):
            raise KeyError
        else:
            alpha, beta = values
        return alpha, beta


class DataConnection:
    def __init__(self, filename: str, config_object: str=None) -> None:
        self.filename = filename
        assert os.path.isfile(filename), "%s is not a valid file" % filename
        self.f = h5py.File(self.filename, "r")
        self.xs = []  # type: List
        if config_object is not None:
            self.cfg = load_config(config_object)
        self.alphas_betas = AlphaBetaFetcher(self.f)
        self.gene_set = set(self.f["genes"].keys())

    def bind(self, config_object: ReconstructionConfig) -> None:
        """Bind the data connection to a config obect
        """
        assert isinstance(config_object, ReconstructionConfig), "DataConnection can only bind to a ReconstructionConfig object"
        self.cfg = config_object

        self.xs = []
        for angle in self.cfg.angles_names:
            self.xs.append(self.f["coordinates/%s" % angle][:])
    
    def __getitem__(self, gene: str) -> List[np.ndarray]:
        return [self.f["genes/%s/%s" % (gene, angle)][:] for angle in self.cfg.angles_names]

    def close(self) -> None:
        self.f.close()
    

def connect_to_data(filename: str, config_object: str=None) -> DataConnection:
    return DataConnection(filename, config_object)
