import os, sys

import numpy as np

from scipy.sparse import diags
from scipy.linalg import inv, svd, svdvals

from tqdm import tqdm, trange

import h5py

apply_mass_matrix = lambda X, M : M.sqrt() @ X
undo_mass_matrix = lambda X, M : M.power(-0.5) @ X

def load_group(f, grpname, rank=6):
    # --> Open the HDF5 group.
    grp = f[grpname]

    # --> Get the POD modes.
    Ψ = grp["POD modes"][:, :rank]

    # --> Extract the mass matrix.
    M = diags(grp["Mass matrix"][:])

    # --> Extract the Reynolds number.
    Re = grp.attrs["Reynolds number"]

    return Ψ, Re, M

def load_dataset(fname):
    # -->
    with h5py.File(fname, "r") as f:

        # --> Load the various POD bases.
        Ψs, Re = list(), list()
        for key in tqdm(f.keys()):
            Ψ, re, M = load_group(f, key)
            Ψs.append(Ψ), Re.append(re)

    Ψs, Re = np.array(Ψs), np.array(Re)

    # --> Re-order the sets with increasing Reynolds numbers.
    idx = np.argsort(Re)
    Ψs, Re = Ψs[idx], Re[idx]

    return Ψs, Re, M

def principal_angles(X, Y):
    # --> Cross-product matrix.
    C = X.T @ Y

    # --> Principal angles.
    σ = svdvals(C)
    θ = np.arccos(σ)

    return θ

def geodesic_distance(X, Y):
    θ = principal_angles(X, Y)
    d = np.sqrt( np.mean(θ**2) )
    return d

def main(ϵ = np.pi/6):

    # --> Load the current database.
    Ψs, Re, M = load_dataset("cylinder_dataset.hdf5")

    # --> Apply the mass matrix to convert to Euclidean norm.
    for i in range(len(Ψs)):
        Ψs[i] = apply_mass_matrix(Ψs[i], M)

    # --> Compute the geodesic length of each cell.
    d = [geodesic_distance(Ψs[i], Ψs[i+1]) for i in range(len(Re)-1)]

    # --> Check if the maximum length is below our tolerance.
    if d.max() < ϵ:
        os.system("touch ./converged")
        sys.exit(0)
    else:
        # --> Search for the cell with the maximum geodesic length.
        idx = np.argmax(d)
        Re_new = 0.5 * (Re[idx] + Re[idx+1])
        return Re_new

if __name__ == "__main__":
    output = main()
