import copy
import os

from rdkit import Chem
from scipy.stats import gaussian_kde
from rdkit.Chem import rdDepictor
from matplotlib import pyplot as plt
rdDepictor.SetPreferCoordGen(True)
from rdkit import RDLogger
from scipy.spatial.transform import Rotation
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import rdMolAlign
from rdkit.Chem import rdDepictor
from rdkit.Geometry import Point3D
rdDepictor.SetPreferCoordGen(True)
from rdkit.Chem import rdMolDescriptors

import numpy as np
from rdkit.Chem import rdDepictor

rdDepictor.SetPreferCoordGen(True)

from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
from matplotlib import pyplot as plt
from scipy import ndimage



def join_molecules(m1, a1, m2, a2):
    """
    Join two molecules by forming a single bond between atom a1 in m1 and atom a2 in m2.

    Parameters:
    - m1: RDKit molecule object for the first molecule
    - a1: Index of the atom in m1 to form the bond
    - m2: RDKit molecule object for the second molecule
    - a2: Index of the atom in m2 to form the bond

    Returns:
    - RDKit molecule object representing the joined molecule
    """
    # Copy molecules to avoid modifying the original ones
    mol1 = Chem.Mol(m1)
    mol2 = Chem.Mol(m2)

    # Add a hydrogen to each atom to create the bond
    mol1 = AllChem.AddHs(mol1)
    mol2 = AllChem.AddHs(mol2)

    # Get the atom indices in the new molecules
    a1_new = a1 + mol1.GetNumAtoms()
    a2_new = a2 + mol1.GetNumAtoms()

    # Add a bond between the specified atoms
    mol1.AddBond(a1, a1_new, Chem.BondType.SINGLE)
    mol2.AddBond(a2, a2_new, Chem.BondType.SINGLE)

    # Combine the two molecules
    combined_mol = Chem.CombineMols(mol1, mol2)

    return combined_mol


def find_2d_local_minima(arr):
    local_minima = []
    rows, cols = arr.shape

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if (arr[i, j] < arr[i - 1, j] and
                    arr[i, j] < arr[i + 1, j] and
                    arr[i, j] < arr[i, j - 1] and
                    arr[i, j] < arr[i, j + 1]):
                local_minima.append((i, j))

    return local_minima


mol = Chem.MolFromMolFile("/Users/alexanderhowarth/Desktop/Projects/AR/EPI-7386_core_R.sdf")
writer = Chem.SDWriter("/Users/alexanderhowarth/Desktop/Projects/AR/EPI-7386_core_R_confs.sdf")


param = rdDistGeom.ETKDGv3()

# Generate conformers
param.pruneRmsThresh = 0.5
param.useRandomCoords = True
param.enforceChirality = True
param.maxAttempts=50

cids = rdDistGeom.EmbedMultipleConfs(mol, 1, param)

AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0, mmffVariant='MMFF94s')

ring_1 = [2,3,5,7,6,4]

ring_2 = [17,14,16,12,9,13,15,18,20]

angles_1 = [ 0, - np.pi/6 , np.pi/6 , np.pi/2 ,-np.pi/2   ]

angles_2 = [ 0, - np.pi/6 , - 5* np.pi/6 , np.pi/6  , 5*np.pi/6, np.pi/2  , - np.pi/2 ,np.pi   ]

pos = mol.GetConformer(0).GetPositions()

ring_1_p = np.array(pos)[ring_1,:]
ring_2_p = np.array(pos)[ring_2,:]

r1_com = mol.GetConformer(0).GetPositions()[2]
r2_com = mol.GetConformer(0).GetPositions()[17]

axis_1 = mol.GetConformer(0).GetPositions()[2] - mol.GetConformer(0).GetPositions()[7]
axis_2 = mol.GetConformer(0).GetPositions()[17] - mol.GetConformer(0).GetPositions()[9]

axis_1 = axis_1 / np.linalg.norm(axis_1, keepdims=True)
axis_2 = axis_2 / np.linalg.norm(axis_2, keepdims=True)

for i , ag1 in enumerate(angles_1):


    for j , ag2 in enumerate(angles_2):

        mol_ = copy.copy(mol)

        conf = mol_.GetConformer(0)

        ring_1_p_c = copy.copy(ring_1_p) - r1_com

        r_1 = Rotation.from_rotvec(ag1 * axis_1)

        ring_1_p_c = copy.copy(ring_1_p_c)
        r1_rotated = r_1.apply(ring_1_p_c)

        ring_2_p_c = copy.copy(ring_2_p) - r2_com

        r_2 = Rotation.from_rotvec(ag2 * axis_2)

        ring_2_p_c = copy.copy(ring_2_p_c)
        r2_rotated = r_2.apply(ring_2_p_c)



        for a1, r_ in zip(ring_1, r1_rotated):
            x, y, z = r_ + r1_com
            conf.SetAtomPosition(a1, Point3D(x, y, z))

        for a2, r_ in zip(ring_2, r2_rotated):
            x, y, z = r_ + r2_com
            conf.SetAtomPosition(a2, Point3D(x, y, z))


        mol_.SetProp("p1" , str(ag1))
        mol_.SetProp("p2" , str(ag2))
        writer.write(mol_,confId = 0)



energies -= np.min(energies)

print(np.max(energies))
print(np.min(energies))

plt.imshow(energies)

plt.ylabel("Ring 1")
plt.xlabel("Ring 2")
plt.close()
plt.plot(angles/np.pi , np.mean(energies,axis = 0))
plt.plot(angles/np.pi , np.mean(energies,axis = 1))


plt.show()

# Find the local minima using ndimage

e_diff1 = np.diff(energies, n=1, axis=0)
e_diff2 = np.diff(energies, n=1, axis=1)



inds = find_2d_local_minima(energies)

plt.show()



'''

#f = Chem.SDMolSupplier("/Users/alexanderhowarth/Desktop/Projects/YTHDC1/YTHDC1_HTRF_18_model/Design_sets")
w = Chem.SDWriter("/Users/alexanderhowarth/Desktop/Projects/YTHDC1/YTHDC1_HTRF_18_model/all_designs.sdf")

for f in os.listdir("/Users/alexanderhowarth/Desktop/Projects/YTHDC1/YTHDC1_HTRF_18_model/Design_sets"):

    if f.endswith(".sdf"):
        sup = Chem.SDMolSupplier("/Users/alexanderhowarth/Desktop/Projects/YTHDC1/YTHDC1_HTRF_18_model/Design_sets/" + f)

        for m in sup:

            if m:
                m.SetProp("file", f)
                w.write(m)

'''