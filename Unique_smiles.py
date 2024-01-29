from rdkit.Chem.Scaffolds import MurckoScaffold
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor
import random
import pickle
rdDepictor.SetPreferCoordGen(True)
import numpy as np
from rdkit.Chem import AllChem
from rdkit import RDLogger
from matplotlib import pyplot as plt
from rdkit.Chem.MolStandardize import rdMolStandardize
import tqdm
from scipy.stats import gaussian_kde
from rdkit.Chem import Descriptors
from rdkit.Chem import Crippen
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.SimDivFilters import rdSimDivPickers
from rdkit.Chem import PandasTools

def standardize(mol):

    clean_mol = rdMolStandardize.Cleanup(mol)
    parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
    uncharger = rdMolStandardize.Uncharger()  # annoying, but necessary as no convenience method exists
    uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
    te = rdMolStandardize.TautomerEnumerator()  # idem
    taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)

    return taut_uncharged_parent_clean_mol

Mols = {}



for s in tqdm.tqdm(open( "/Users/alexanderhowarth/Desktop/52039_test_model_2/sampled/sample_model2_990.smi", "r").readlines()):

    print(s)

    m = Chem.MolFromSmiles(s)

    if m:

        Mols[Chem.MolToInchiKey(m)] = [ s ,MurckoScaffold.MurckoScaffoldSmilesFromSmiles(s)]




print(len(Mols))

pickle.dump(Mols,open("/Users/alexanderhowarth/Desktop/52039_test_model_2/Mols.p","wb"))



Mols = pickle.load(open("/Users/alexanderhowarth/Desktop/52039_test_model_2/Mols.p","rb"))

w = Chem.SDWriter("/Users/alexanderhowarth/Desktop/52039_test_model_2/Mols.sdf")

qp = Chem.AdjustQueryParameters()
qp.makeDummiesQueries = True
qp.adjustDegree = True
qp.adjustDegreeFlags = Chem.ADJUST_IGNOREDUMMIES


core_ = "CNc1ncnc2c(*)n[nH]c12"

core_ = Chem.MolFromSmiles(core_)
qm = Chem.AdjustQueryProperties(core_, qp)

for m_ in Mols.values():

    m = Chem.MolFromSmiles(m_[0])

    if m:

        if m.HasSubstructMatch(qm):

            m.SetProp("Murcko" , str(m_[1]))

            w.write(m)