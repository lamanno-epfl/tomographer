Summary
========

The Tomographer package implements the mathematical spatial image reconstruction of any set of read counts as described in this paper (reference). Tutorials are provided to facilitate its implementation.

Section 1: Required Inputs
----------------------------

Tomographer requires a number of inputs in the correct format. 

* **The design matrix**. The design matrix encodes information regarding the experimental setup parameters. These parameters are listed in the class `ReconstructionConfig` (refer to documentation of `defaults.py`) as arguments. The function will create an `.hdf5` formatted file that will be used as the configuration file.

* **The projection data**. The input genes are also stored in an `.hdf5` file in the form of their projection data.

* **The list of genes to reconstruct**. This is simply `.txt` file which lists line by line the gene names from the projection data one wishes to reconstruct. Each gene must be separated by `\n`.

Section 2: Reconstruction Parameters
--------------------------------------

There are a number of parameters that can be modified in this package. Depending on the precise application, the precise parameters may change. `tomorun.py` must be modified in function `reconstruction` to reflect these changes.

* **Solver**. The default solver uses an optimization package from Scipy due to its speed. The solver can be switched to cvxpy (ReconstructorCVXPY), which in some cases produces better results.

* **Hyperparameter search**. A number of parameters are related to the search. (1) The range for the parameters and resolution of the grid search can be specified. If one already has an idea of what hyperparameters one wants to use, they can be specified directly and no search will be performed. (2) Using a logged grid. The default is a linear grid. (3) Number of extra evaluations to be performed using Bayesian optimization after grid search. Default is 7.



Section 3: Outputs
-----------------------

The output of running Tomographer are two `.hdf5` files:

* **Reconstructed genes**. These are the images after mathematical reconstruction. Gene names are listed as keys.

* **Alpha beta parameters**. If Tomographer was asked to find the optimal alpha-beta hyperparameters, the selected parameters are stored in this `.hdf5` file. This can be useful for further analysis for filtering genes.

Section 4: Commands for reconstruction
------------------------------------------

The file `tomorun.py` takes in 5 arguments and will perform the entire pipeline. 

The 5 arguments are the following:

* **-c** configurationFile. This is the config file that was generated to specify experimental parameters

* **-i** inputProjections.hdf5. This contains the projection data

* **-o** outputFileName. This is the name you want to give to your output file

* **-a** outputFileNameAlphaBeta. This is the name you want to give to your alpha-beta output file.

* **-g** listofGenes. This is the text file containing gene names you wish to reconstruct

An example usage of `tomorun.py` would be ::

    export OMP_NUM_THREADS && OMP_NUM_THREADS=2 && nohup python3 /tomography/tomorun.py -c /configurationFile.hdf5 -i /inputProjections.hdf5 -o /outputFileName.hdf5 -a  outputFileNameAlphaBeta.hdf5 -g listofGenes.txt> /outputstderr.txt 2>&1 &

Note that the first line `export OMP_NUM_THREADS && OMP_NUM_THREADS=2 && nohup` can be useful for preventing all cores to be used simultaneously. In this case the number of cores are restricted to 2, but this can be changed. 