import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import h5py
from typing import *
import logging
from tomography import build_Design_Matrix, prepare_regression_symmetry, prepare_regression_symmetry_masked
from tomography import ReconstructorFastScipyNB, mixing_with_img
from tomography import Tomographer
import sys
import argparse
import time
import os


class MyParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)


class Clock:
    def __init__(self) -> None:
        self.internal = 0.

    def tic(self) -> None:
        self.internal = time.time()
    
    def toc(self) -> float:
        return time.time() - self.internal

    def reset(self) -> None:
        self.internal = 0


if __name__ == "__main__":
    parser = MyParser(description='Reconstruct Spatial gene expression')
    parser.add_argument('-c', '--config', metavar='config', required=True, type=str, help='File hdf5 with reconstruction config hdf5')
    parser.add_argument('-i', '--input', metavar='input', required=True, type=str, help='File hdf5 with values from the genes')
    parser.add_argument('-o', '--output', metavar='output', required=True, type=str, help='Filename of the hdf5 output file')
    parser.add_argument('-a', '--al_be', metavar='alpha_beta', required=True, type=str, help='Filename of the hdf5 alpha_beta output file')
    parser.add_argument('-g', '--genes', metavar='genes', required=False, type=str, help='string of space separated gene names')
    parser.add_argument('-t', '--test', action='store_true', help='Run in testmode ignoring the input and runnin 10 genes')
    args = parser.parse_args()

    # Take care of the output
    logging.basicConfig(stream=sys.stdout, format='%(asctime)s - PID:' + str(os.getpid()) + ' - %(levelname)s - %(message)s', level=logging.DEBUG)   

    # N-process and output file name
    outfilename = args.output
    alpha_beta_file = args.al_be

    if args.genes is not None:
        tg = Tomographer()
        tg.load_cfg(args.config)
        tg.connect_data(args.input)
        available = list(tg.data.f["genes"].keys())
        genes_to_reconstruct = open(args.genes).read().rstrip().split()
        genes_to_reconstruct = [i for i in genes_to_reconstruct if i in available]
        del tg
        logging.debug(f"{len(genes_to_reconstruct)} genes to reconstruct {', '.join(genes_to_reconstruct[:50])} ...")
    else:
        tg = Tomographer()
        tg.load_cfg(args.config)
        tg.connect_data(args.input)
        genes_to_reconstruct = list(tg.data.f["genes"].keys())
        del tg

    logging.debug("==> Results will be saved to %s <==" % outfilename)

    def reconstruction(gene: str) -> Tuple[str, Tuple, np.ndarray]:
        logging.debug("Spawn a tomographer")
        tg = Tomographer()
        tg.load_cfg(args.config)
        tg.connect_data(args.input)
        logging.debug("Reconstructing %s started ..." % gene)
        ck = Clock()
        ck.tic()
        reconstructed_img = tg.reconstruct(gene, 'auto', 
                                           crossval_kwargs={'domain': [(0,4), (0.05,8)],
                                                            'logged_grid':False, 
                                                            'extra_evals': 7,
                                                            'style' : 'grid',
                                                            'gradient_iter' : 5,
                                                            'initial_grid_n': 5})
        logging.debug("Reconstructing of %s finished in : %.1f" % (gene, ck.toc()))
        alpha_beta = tg.reconstructor.alpha, tg.reconstructor.beta
        # logging.debug(f'Gene {gene}, alpha {alpha_beta}, data {reconstructed_img.shape}')
        return gene, alpha_beta, reconstructed_img

    alpha_beta_dict = {}

    def handle_output(name, alpha_beta, data) -> None:
        """Saves the result of reconstructions to two hdf5 files.
        One contains reconstructed genes and the other the chosen alpha beta values.
        If run as a callback will happen in its own independent process.
        """
        
        # name, alpha_beta, data = recontruction_returns
        #alpha_beta_dict[name] = alpha_beta

        if data is None:
            logging.warning("!!! %s was too low, don't reconstruct" % name)
        else:
            # Write files
            fout = h5py.File(outfilename, mode='a')
            f_ab = h5py.File(alpha_beta_file, mode='a')
            fout.create_dataset(name, data=data)
            f_ab.create_dataset(name, data=alpha_beta)
            fout.close()
            f_ab.close()
    
    # clear the file
    if not os.path.isfile(outfilename):
        fout = h5py.File(outfilename, mode='w')
        fout.close()
        f_ab = h5py.File(alpha_beta_file, mode='w')
        f_ab.close()
   
 
    for gene in genes_to_reconstruct:
        # try:
        gene, alpha_beta, data = reconstruction(gene)
        handle_output(gene, alpha_beta, data)
        # except:
            # print (gene, "Passed over *#*#*#")


