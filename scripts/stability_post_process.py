import os

import numpy as np

from scipy.linalg import svd
from scipy.sparse.linalg import svds

from sklearn.utils.extmath import svd_flip

from glob import glob
from tqdm import trange, tqdm
import h5py

from nekio import *

def load_stability_results():
    # -->
    fname = "./data/STABILITY/RevCYL0.f00001"
    ωr = readnek(fname)["data"][:, :, -1].ravel()

    # -->
    fname = "./data/STABILITY/ImvCYL0.f00001"
    ωi = readnek(fname)["data"][:, :, -1].ravel()

    # -->
    fname = "./data/STABILITY/Spectre_NS_conv.dat"
    λr, λi = np.loadtxt(fname, unpack=True)

    return ωr + 1j * ωi, λr[0] + 1j * λi[0]

def load_base_flow():
    # --> Name of the file.
    fname = "./data/STABILITY/BF_CYL0.f00001"

    # --> Base flow vorticity field.
    ω = readnek(fname)["data"][:, :, -1].ravel()

    return ω

def main():

    # --> Get the Reynolds number to process.
    Re = sys.argv[1]

    # --> Load the base flow.
    X̄ = load_base_flow().reshape(-1, 1)

    # --> Load the stability results.
    X, λ = load_stability_results()

    # --> Store the data in the HDF5 archive.
    with h5py.File("cylinder_dataset.hdf5", "a") as f:

        # -->
        grp = require_group("RE{0}".format(Re))
        grp.attrs["Eigenvalue"] = λ

        # --> Store the base flow and stability mode.
        grp.create_dataset("Base flow", data=X̄)
        grp.create_dataset("Stability mode", data=X)

if __name__ == "__main__":
    main()
