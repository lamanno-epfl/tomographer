# Tomographer
Software for spatial tissue profiling by imaging-free molecular tomography.

# Installation

Installation is achieved in two commands:

First clone locally
```bash
git clone https://github.com/lamanno-epfl/tomographer.git
```
and then install in development mode
```bash
pip install -e ./tomographer
```
To update you just need to do (you don't need to reinstall): 
```bash
cd tomographer
git pull
```

## Requirements

This package requires:
* The following common packages (available on conda):
```
'numpy'
'scipy'
'pandas'
'scikit-learn'
'scikit-image'
'matplotlib'
```

* And these less known libraries (to install with pip or by cloning source and python setup install):
```
'gp_extras'
'PyWavelets'
'gpy'
'gpyopt'
```

## Use

The package requires 3 input files and 2 output directories.

### Inputs

- **Config File** : This is an `.hdf5` formatted file which contains the following keys and values
  - _angles_ _ _names_ : array of angle names like ['angle9', 'angle117']
  - _angles_ _ _values_ : array of angle values in radians like [3.65, 5.096]
  - _first_ _ _points_ : array specifying the starting point to begin considering values within projection for each angle
  - _mask_ _ _g_ : 2D image array of mask (binary)
  - _mask_ _ _thrs_ : threshold used to create mask from reference image
  - _masked_ _ _formulation_ : Boolean indicating if design matrix should be recalculated using the masked formulation
  - _proj_ _ _len_ : array indicating the lengths of projections for each angle
  - _reference_ _ _mask_ : image array of reference image
  - _symmetry_ : Boolean indicating if design matrix should be recalculated assuming axis of symmetry in tissue
  - _widths_ : array indicating the estimated widths of the secondary slices in each angle
  - _A_ : Specified design matrix. Note that this matrix is recalculated if the _symmetry_ value is True or if the _masked formulation_ value is True

- **Input File** : This is an `.hdf5` formatted file which contains a dictionary of dictionaries. It can be created with the **ReconstructionConfig** Class 
  - The key *genes* is further queried by
    - The *gene name*, which itself contains the keys containing
      - The *angle names* that match those in the config file
        - Values inside input_file['genes'][gene_name][angle_name] correspond to the projection values

- *Gene File*: Is a `.txt` formatted file which contains the individual gene names separated by lines (`\n`). These are queried by the tomographer one by one

### Outputs

- **Output Reconstruction File** : This is an `.hdf5` formatted file which contains all the reconstructed genes
- **Output Alpha-Beta File** : This is an `.hdf5` formatted file which contains all the selected alpha and beta values that were selected for a given reconstruction. This may be useful for filtering out poorly reconstructed genes.

### Example Usage

From tomographer, one can run the following command:

```python3 tomography/tomorun.py -c path_to_config.hdf5 -i path_to_inputs.hdf5 -g /list_of_genes.txt -o /path_to_output.hdf5 -a  /path_to_/alpha_beta_output.hdf5 ```
