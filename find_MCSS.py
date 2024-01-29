from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Draw
from rdkit.Chem import rdRGroupDecomposition
from rdkit.Chem import rdqueries
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import Geometry
rdDepictor.SetPreferCoordGen(True)
import pandas as pd
from rdkit.Chem import AllChem
from PIL import Image as pilImage
from io import BytesIO
from rdkit.Chem.MolStandardize import rdMolStandardize
import rdkit

from rdkit.Chem import rdFMCS
from collections import defaultdict
from PIL import Image as pilImage
from io import BytesIO
import pickle
from rdkit.Chem import PandasTools


def standardize(mol):

    clean_mol = rdMolStandardize.Cleanup(mol)
    parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
    uncharger = rdMolStandardize.Uncharger()  # annoying, but necessary as no convenience method exists
    uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
    te = rdMolStandardize.TautomerEnumerator()  # idem
    taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)

    return taut_uncharged_parent_clean_mol

rdDepictor.SetPreferCoordGen(True)

output_folder = "3486"

df = PandasTools.LoadSDF("/Users/alexanderhowarth/Desktop/YAP_followup/labeled_MF_quote.sdf")

output_folder = "4618"

Mols_ = [ standardize(m) for m in df.ROMol  ]

ms =[]

import json

series_ = [ s for s in df.series  ]

BOAW_Mols = []
Tan_Mols = []

for s,m in zip(series_,Mols_):

    if output_folder in s:
        ms.append(m)

mcs = rdFMCS.FindMCS(ms, completeRingsOnly=True, ringMatchesRingOnly=True)

mcs_mol = Chem.MolFromSmarts(mcs.smartsString)



AllChem.Compute2DCoords(mcs_mol)

pickle.dump(mcs_mol,open(output_folder + "/MCSS_mol.p","wb"))


