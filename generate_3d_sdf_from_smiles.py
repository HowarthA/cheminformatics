from rdkit  import Chem
import numpy as np
import tqdm

import os
import multiprocessing
from rdkit.Chem import rdmolops
import os.path

from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolAlign
import tqdm
import numpy as np
import multiprocessing as mp
from rdkit import RDLogger


m = Chem.MolFromSmiles("COc2cc(/C=C(C#N)\C(=O)Nc1ccc(Cl)cc1)cc(Cl)c2OCC3CCC(C(=O)O)CC3")

output = "/Users/alexanderhowarth/Desktop/TRB52809_saturated_rdkit_min.sdf"

m = rdmolops.AddHs(m)

param = rdDistGeom.ETKDGv3()

param.pruneRmsThresh = 0.2

n_conformers = 10

cids = rdDistGeom.EmbedMultipleConfs(m, 1, param)

AllChem.MMFFOptimizeMoleculeConfs(m, numThreads=0, mmffVariant='MMFF94s')

writer =Chem.SDWriter(output )

writer.write(m)