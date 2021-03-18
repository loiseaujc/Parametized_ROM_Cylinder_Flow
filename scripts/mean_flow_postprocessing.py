import os

import numpy as np

from scipy.linalg import svd
from scipy.sparse.linalg import svds

from sklearn.utils.extmath import svd_flip

from glob import glob
from tqdm import trange, tqdm
import h5py

from nekio import *

apply_mass_matrix = lambda X, M : M.sqrt() @ X
undo_mass_matrix = lambda X, M : M.power(-0.5) @ X

def pod(X, rank=6):
    # --> Compute the rank-k truncated SVD of X.
    U, Σ, Vh = svds(X, k=rank)

    # --> ARPACK does not abide by SVD convention.
    idx = np.argsort(-Σ)
    Σ = Σ[idx]

    # --> Sign correction to ensure deterministic output from SVD.
    U, Vh = svd_flip(U[:, idx], Vh[idx])

    # --> Low-dimensional PCA state vector.
    a = np.diag(Σ) @ Vh

    return U, a, Σ**2

def load_snapshots():

    # --> Directory name.
    dname = "./data/MEAN_FLOW/".format(Re)

    # --> Load the vorticity field.
    X = np.array([
        readnek(f)["data"][:, :, -1].ravel() for f in tqdm(glob(dname + "CYL0.f*"))
        ]).T

    # --> Clean-up the simulation.
    for f in sorted(glob(dname + "CYL0.f*")):
        os.remove(f)

    return X

def load_mesh_and_mass_matrix():

    # --> Name of the file.
    fname = "./data/MEAN_FLOW/BM1CYL0.f00001"

    # --> Load data.
    data = readnek(fname)["data"]

    # --> Extract mesh and mass matrix.
    x, y = data[:, :, 0].ravel(), data[:, :, 1].ravel()
    M = data[:, :, 2].ravel()

    return np.c_[x, y], M

def load_mean_flow():

    # --> Name of the file.
    fname = "./data/MEAN_FLOW/avgCYL0.f00001"

    # --> Mean flow vorticity field.
    ω = readnek(fname)["data"][:, :, -1].ravel()

    return ω

def main():

    # --> Get the Reynolds number to process.
    Re = sys.argv[1]

    # --> Load the mesh and mass matrix.
    mesh, M = load_mesh_and_mass_matrix()

    # --> Load the mean flow.
    X̄ = load_mean_flow().reshape(-1, 1)

    # --> Load the snapshots.
    X = load_snapshots()

    # --> Mean center the data.
    X -= X̄

    # --> Perform POD analysis.
    X = apply_mass_matrix(X, M)
    U, a, Λ = pod(X)
    U = undo_mass_matrix(U, M)

    # --> Store the data in the HDF5 archive.
    with h5py.File("cylinder_dataset.hdf5", "a") as f:

        # --> Create dedicated group.
        grp = f.require_group("RE{0}".format(Re))
        grp.attrs["Reynolds number"] = np.float(Re)

        # --> Store the data.
        grp.create_dataset("Mesh", data=mesh)
        grp.create_dataset("Mass matrix", data=M)
        grp.create_dataset("Mean flow", data=X̄)
        grp.create_dataset("POD modes", data=U)
        grp.create_dataset("POD eigenvalues", data=Λ)
        grp.create_dataset("POD amplitudes", data=a)

if __name__ == "__main__":
    main()
