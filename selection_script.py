import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import DataStructs
import random
import pickle
from scipy.stats import gaussian_kde
import tqdm
from rdkit.Chem import PandasTools
import copy
import pandas as pd

hit_df = PandasTools.LoadSDF("/Users/alexanderhowarth/Desktop/Projects/EML4-ALK/Further_hits/hits.sdf")


#rd_df = PandasTools.LoadSDF("/Users/alexanderhowarth/Desktop/Projects/EML4-ALK/Further_hits/All_mols_sol_filter.sdf",embedProps=True)


rd_df = PandasTools.LoadSDF("/Users/alexanderhowarth/Desktop/Projects/EML4-ALK/Further_hits/TRB0031782_Mcule.sdf",embedProps=True)

IDs = [ m.GetProp("mcule ID") for m in Chem.SDMolSupplier("/Users/alexanderhowarth/Desktop/Projects/EML4-ALK/Further_hits/TRB0031782_Mcule.sdf") if m ]


rd_df["ID"] = IDs

'''TRBs= [

"TRB0030216",
"TRB0029784",
"TRB0029974",
"TRB0029621",
"TRB0031782",
"TRB0035815",
"TRB0022836",
"TRB0030001",

]'''

ID= "TRB0031782"


hit_mol_df = hit_df[ hit_df["FORMATTED_ID" ]==ID ].iloc[0]

hit_mol = hit_mol_df["ROMol"]

hit_fp = AllChem.GetMorganFingerprintAsBitVect(hit_mol,3)

mol_select = []




fps = []

for m in rd_df.ROMol:
    fps.append(AllChem.GetMorganFingerprintAsBitVect(m,3))

sims = DataStructs.BulkTanimotoSimilarity( hit_fp, fps )

rd_df["sims"] = sims


selection = pd.DataFrame(columns=[c for c in rd_df.columns])

rd_df = rd_df[rd_df["sims"] > 0.3]

mods = set(rd_df["select"])

for m in mods:

    m_df = rd_df[rd_df['select'] == m]

    print(m, min(12, len(m_df)))

    m_df = m_df.sort_values(by=['sims'], ascending=False)

    selection = pd.concat([selection, m_df.iloc[0:min(10, len(m_df))]])

PandasTools.WriteSDF(selection,
                     "/Users/alexanderhowarth/Desktop/Projects/EML4-ALK/Further_hits/selections/" + ID + "_Mcule_selection.sdf",
                     properties=[c for c in rd_df.columns])

quit()



s  = 0

total_selection = None

for ID in TRBs:

    hit_mol_df = hit_df[ hit_df["FORMATTED_ID" ]==ID ].iloc[0]

    hit_mol = hit_mol_df["ROMol"]

    hit_fp = AllChem.GetMorganFingerprintAsBitVect(hit_mol,3)

    mol_select = []

    ID_df = copy.copy(rd_df[rd_df[ "TRB"  ] ==ID])

    fps = []

    for m in ID_df.ROMol:
        fps.append(AllChem.GetMorganFingerprintAsBitVect(m,3))

    sims = DataStructs.BulkTanimotoSimilarity( hit_fp, fps )

    Stock = []

    for i,r in ID_df.iterrows():

        if "Stock" in r['file']:

            Stock.append(1)

        else:

            Stock.append(0)

    mods = set(ID_df["select"])

    ID_df["Stock"] = Stock

    ID_df["sims"] = sims

    stock = ID_df[ID_df['Stock'] == 1]

    ID_df["Stock"] = Stock

    ID_df["sims"] = sims

    PandasTools.WriteSDF(ID_df,"/Users/alexanderhowarth/Desktop/Projects/EML4-ALK/Further_hits/selections/" + ID + ".sdf",properties=[c for c in ID_df.columns])

    selection = pd.DataFrame(columns=[c for c in ID_df.columns])

    if total_selection is None:

        total_selection =pd.DataFrame(columns=[c for c in ID_df.columns])


    if ID not in ["TRB0022836", "TRB0030001"]:

        ID_df = ID_df[ID_df["sims"] > 0.3]

        print("")
        print("ID",ID)
        for m in mods:
            if m not in ["","other","none"]:

                m_df = ID_df[ID_df['select'] == m]

                print(m , min(10, len(m_df)))

                m_df = m_df.sort_values(by=['sims'],ascending=False)

                selection = pd.concat([selection , m_df.iloc[0:min(10,len(m_df))]])

        if len(selection) < 50:
            print("other " , 50 - len(selection))

            m_df = ID_df[(ID_df['select'] == "none") | (ID_df['select'] == "other") | (ID_df['select'] == "") ]

            m_df = m_df.sort_values(by=['sims'], ascending=False)

            selection = pd.concat([selection, m_df.iloc[0:( 50 - len(selection))]])


        total_selection = pd.concat([total_selection, selection])
        PandasTools.WriteSDF(selection,"/Users/alexanderhowarth/Desktop/Projects/EML4-ALK/Further_hits/selections/" + ID + "_selection.sdf",properties=[c for c in ID_df.columns])

    if ID == "TRB0022836":

        ID_df = ID_df[ID_df["sims"] > 0.3]
        print("")
        print("ID",ID)

        ID_df = ID_df[ID_df['Stock'] == 1]

        for m in mods:
            if m not in ["", "other", "none"]:

                m_df = ID_df[ID_df['select'] == m]
                print(m , min(10, len(m_df)))

                m_df = m_df.sort_values(by=['sims'], ascending=False)

                selection = pd.concat([selection, m_df.iloc[0:min(10, len(m_df))]])

        if len(selection) < 50:
            print("other " , 50 - len(selection))
            m_df = ID_df[(ID_df['select'] == "none") | (ID_df['select'] == "other") | (ID_df['select'] == "")]

            m_df = m_df.sort_values(by=['sims'], ascending=False)

            selection = pd.concat([selection, m_df.iloc[0:(50 - len(selection))]])

        PandasTools.WriteSDF(selection,
                             "/Users/alexanderhowarth/Desktop/Projects/EML4-ALK/Further_hits/selections/" + ID + "_selection.sdf",
                             properties=[c for c in ID_df.columns])

    if ID == "TRB0030001":

        PandasTools.WriteSDF(ID_df,
                             "/Users/alexanderhowarth/Desktop/Projects/EML4-ALK/Further_hits/selections/" + ID + "_selection.sdf",
                             properties=[c for c in ID_df.columns])




PandasTools.WriteSDF(total_selection,"/Users/alexanderhowarth/Desktop/Projects/EML4-ALK/Further_hits/selections/total_selection.sdf",properties=[c for c in ID_df.columns])
