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
import pickle
import rdkit

from rdkit.Chem import rdFMCS
from collections import defaultdict
from PIL import Image as pilImage
from io import BytesIO

from rdkit.Chem import PandasTools
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

rdDepictor.SetPreferCoordGen(True)

def standardize(mol):

    clean_mol = rdMolStandardize.Cleanup(mol)
    parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
    uncharger = rdMolStandardize.Uncharger()  # annoying, but necessary as no convenience method exists
    uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
    te = rdMolStandardize.TautomerEnumerator()  # idem
    taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)

    return taut_uncharged_parent_clean_mol

df = PandasTools.LoadSDF("/Users/alexanderhowarth/Desktop/EML4-quotes/labeled_full_quote.sdf")

full_TRB = 'TRB0003486'

code = "3486"

output_folder = "/Users/alexanderhowarth/Desktop/YAP_followup/3486"

Mols_ = [ standardize(m) for m in df.ROMol  ]

series_ = [ s for s in df.series  ]

BOAW_Mols = []
Tan_Mols = []

for s,m in zip(series_,Mols_):

    if code in s:

        if "BOAW" in s:

            BOAW_Mols.append(m)

        else:

            Tan_Mols.append(m)

BOAW_Murcko =[]
Tan_Murcko =[]

for m in BOAW_Mols:

    BOAW_Murcko.append( MurckoScaffold.MurckoScaffoldSmilesFromSmiles(Chem.MolToSmiles(m)))

BOAW_Murcko = list(set(BOAW_Murcko))

print(len(BOAW_Mols))

print(len(BOAW_Murcko))

quit()

BOAW_Murcko = [  Chem.MolFromSmiles(m) for m in BOAW_Murcko ]

img = Draw.MolsToGridImage(BOAW_Murcko, subImgSize=(250,250),molsPerRow=min(10,len(BOAW_Murcko)))

img.save( output_folder +'/Fuzzy_Pharmacophore_Murcko.png')

for m in Tan_Mols:

    Tan_Murcko.append( MurckoScaffold.MurckoScaffoldSmilesFromSmiles(Chem.MolToSmiles(m)))

Tan_Murcko = list(set(Tan_Murcko))

print(Tan_Murcko)

Tan_Murcko = [  Chem.MolFromSmiles(m) for m in Tan_Murcko ]


MCSS_mol = pickle.load( open(output_folder + "/MCSS_mol.p" ,"rb"))

for m in Tan_Murcko:

    _ = AllChem.GenerateDepictionMatching2DStructure(m,MCSS_mol,allowRGroups= True)

    print(_)

img = Draw.MolsToGridImage(Tan_Murcko, subImgSize=(250,250),molsPerRow=min(3,len(Tan_Murcko)))

img.save( output_folder +'/Tanimoto_Murcko.png')

