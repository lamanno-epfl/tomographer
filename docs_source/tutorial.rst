Tutorials
=============

Create configuration and input files
----------------------------------------

The configuration file needs to be created which will incorporate all experimental set up parameters. Please refer to documentation of defaults.ReconstructionConfig()::

    configurationFile = tomography.ReconstructionConfig(angles_names=angles_names,
                                           mask=mask,mask_thrs=0.1,
                                           reference_mask=reference_mask,
                                           symmetry=True,
                                           masked_formulation=True,
                                           angles_values=angles_values,
                                           first_points=first_points,
                                           widths=widths)

The mask is a binarized image that indicates which pixels are located within the tissue with 1 and the pixels outside the tissue with 0. The reference mask is a brightfield microscopy image of the tissue which was thresholded by the parameter mask_thrs to create the binarized mask. The symmetry parameter implies that one is considering a tissue section which is believed to be symmetrical. Setting this to true permits for more accurate reconstructions for symmetrical tissue samples. 

The input file needs to be in a specified format as well. It is a dictionary of dictionaries. Depending on the data you have as projections, you will need to modify the line `gene_projection = load(gene, angle)` so that the data can be properly loaded and stored in the `.hdf5` file.::

    def create_filename_data(file_path, angles_names, angles_values, var):
        """Creates h5py file with data.
        
        Argumets:
        
        file_path: string path to .h5py
        angles_names: list of angle names in format anglexxx
        angles_values: integer values of angle names
        var: dictionary containing gene_list and attributes
        """
        
        fout = h5py.File(file_path, 'w')
   
        for g_name, g_ in var.items():
            
            for i, anglename in enumerate(angles_names):
                
                gene_projection = load(gene, angle)
                
                fout.create_dataset("genes/%s/%s" % (g_name, anglename), data=gene_projection)

        fout.close()
        return profiles

The Tomographer object
------------------------

In the following steps, we will create the Tomographer object and load the ReconstructionConfig file and InputProjection file.

This is done as follows::

    filename_data = "/path/to/InputProjection.hdf")
    tg = tomography.Tomographer()
    tg.load_cfg(configurationFile)
    tg.connect_data(filename_data)

Next, the Tomographer object has a method `reconstruct`. This will take in the gene name and the hyperparameters one wants to use to solve for the spatial profile of the gene.::

    result = tg.reconstruct(gene, alpha_beta=(0,0)) 

If one is unsure of the hyperparameters, replace (0,0) with 'auto' to perform a grid search.::

     result = tg.reconstruct(gene, crossval_kwargs={"domain":[(-5, 0.8), (-6, 2.5)]})

Note that other specifications for the grid search can be found in optimization.optimize. 
    


Reconstruct multiple genes
-----------------------------

If one has a list containing multiple gene names which one would like to reconstruct, `tomorun.py` can be run using the inputs specified in summary (section 4)::

    export OMP_NUM_THREADS && OMP_NUM_THREADS=2 && nohup python3 /tomography/tomorun.py -c /configurationFile.hdf5 -i /inputProjections.hdf5 -o /outputFileName.hdf5 -a  outputFileNameAlphaBeta.hdf5 -g listofGenes.txt> /outputstderr.txt 2>&1 &
