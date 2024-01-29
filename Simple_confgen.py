import os.path

from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolAlign
import tqdm
import numpy as np
import multiprocessing as mp
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def embed_mol(mol):

    param = rdDistGeom.ETKDGv3()

    param.pruneRmsThresh = 0.2

    n_conformers = 10

    cids = rdDistGeom.EmbedMultipleConfs(mol, n_conformers, param)

    AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0, mmffVariant='MMFF94s')

    Chem.rdMolAlign.AlignMolConformers(mol)

    return mol, cids


param = rdDistGeom.ETKDGv3()

param.pruneRmsThresh = 0.2

n_conformers = 10


w = Chem.SDWriter("/Users/alexanderhowarth/Desktop/YTHDC1_REAL/DR_linear_gradients_structures_rdconf.sdf")

for mol in Chem.SDMolSupplier("/Users/alexanderhowarth/Desktop/YTHDC1_REAL/DR_linear_gradients_structures_proton.sdf",removeHs = False):

    cids = rdDistGeom.EmbedMultipleConfs(mol, n_conformers, param)

    AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0, mmffVariant='MMFF94s')

    Chem.rdMolAlign.AlignMolConformers(mol)

    for c in cids:

        w.write(mol,c)