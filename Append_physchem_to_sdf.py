import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor
import random
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

lp_minmax = rdSimDivPickers.MaxMinPicker()
lp_leader = rdSimDivPickers.LeaderPicker()

df = PandasTools.LoadSDF("/Users/alexanderhowarth/Desktop/EML4-quotes/labeled_full_quote.sdf")

def get_props_2d_continious(mols):

    mw =[ ]

    TPSA = []

    logp = []
    fsp3 = []

    for m in mols:

        mw.append(Descriptors.MolWt(m))

        TPSA.append( rdMolDescriptors.CalcTPSA(m))

        logp.append(Crippen.MolLogP(m))
        fsp3.append(rdMolDescriptors.CalcFractionCSP3(m))

    return mw,TPSA,logp,fsp3

mw,TPSA,logp,fsp3 = get_props_2d_continious(df.ROMol)


df['MW'] = mw

df['TPSA'] = TPSA

df['cLogp'] = logp

df['fraction sp3'] = fsp3

PandasTools.WriteSDF(df,"labeled_selection.sdf",properties=list(df.columns))