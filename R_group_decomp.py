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
from rdkit.Chem import rdmolops
import rdkit

from rdkit.Chem import rdFMCS
from collections import defaultdict
from PIL import Image as pilImage
from io import BytesIO
import pickle
from rdkit.Chem import PandasTools
from rdkit.Chem.Scaffolds import MurckoScaffold
import copy
from rdkit.Chem.MolStandardize import rdMolStandardize

'''
from rdkit.Chem import rdRGroupDecomposition
s = Chem.MolFromSmarts('*Nc1ccc(*)cc1')
ms = list(map(Chem.MolFromSmiles, ['CC(=O)Nc1ccc(O)cc1','CCNc1ccc(OC)cc1']))
gs, _ = rdRGroupDecomposition.RGroupDecompose([s], ms)
'''


def standardize(mol):
    clean_mol = rdMolStandardize.Cleanup(mol)
    parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
    uncharger = rdMolStandardize.Uncharger()  # annoying, but necessary as no convenience method exists
    uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
    te = rdMolStandardize.TautomerEnumerator()  # idem
    taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)

    return taut_uncharged_parent_clean_mol


def draw(mols, labels, file):
    from rdkit.Chem.Draw import rdMolDraw2D

    n_per_row = min(4, len(mols))

    scale = 300

    n_rows = int(len(mols) / n_per_row) + (len(mols) % n_per_row > 0)

    d2d = rdMolDraw2D.MolDraw2DSVG(n_per_row * scale, n_rows * scale, scale, scale)

    d2d.DrawMolecules(list(mols[0:100]), legends=labels)

    pic = open(file + ".svg", "w+")

    d2d.FinishDrawing()

    pic.write(str(d2d.GetDrawingText()))

    pic.close()


rdDepictor.SetPreferCoordGen(True)

# core_ = Chem.MolFromSmiles('C(=C/c2ccc(OCc1ccccc1)cc2)C(=O)Nc3ccccc3')

# 3486
# hit_mol = Chem.MolFromSmiles('Cc1cccc(-c2nc(CCCC3)c3c(SCC(O)=O)n2)c1')

# 1264

# hit_mol =Chem.MolFromSmiles("Cc(c1cccnc11)cc(C(C(F)(F)F)(C(OC)=O)O)c1O")

#hit_smiles = "Fc1ccc(cc1)-c1cnc(CN2C(=O)NC3(CCSC3)C2=O)o1"
hit_smiles = "CCOC1CC(N(C)CC(=O)Nc2c(C)n[nH]c2C)C11CCCCC1"

full_TRB = 'TRB0023566'

output_folder = "/Users/alexanderhowarth/Desktop/Projects/EML4-ALK/EML4-ALK_TL/TRB0023566/Enamine_selections/"

initial_hit = standardize(Chem.MolFromSmiles(hit_smiles))

centre_fp = AllChem.GetMorganFingerprintAsBitVect(initial_hit, 2, 1024)
'''
df = pd.read_csv("/Users/alexanderhowarth/Desktop/Projects/EML4-ALK/EML4-ALK_TL/Quote_1691635_EUR.csv")



ms = []

for i, r in df.iterrows():

    if r['Series'] == full_TRB:
        ms.append(standardize(Chem.MolFromSmiles(r['SMILES'])))

'''


ms = [ m for m in Chem.SDMolSupplier("/Users/alexanderhowarth/Desktop/Projects/EML4-ALK/EML4-ALK_TL/TRB0023566/Enamine_selections/Z1263185411_RDB_3573analogues.sdf") ]

ms = [ ms[i] for i in range(0,min(len(ms ), 50)) ]


Murcko = []

for m in ms:
    Murcko.append(Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m)))

murc_dic = {}

for murk in Murcko:

    if murk in murc_dic.keys():

        murc_dic[murk] +=1

    else:

        murc_dic[murk] = 0

Murcko_list_sm = []

for k,v in zip(murc_dic.keys(),murc_dic.values()):

    if v > 1:

        Murcko_list_sm.append(k)



# Murcko = ["c1ccc(COc2c(OC)cc(/C=C\C(=O)Nc3ccccc3)cc2)cc1"]

# Murcko_list_sm = ["c1ccc(COc2c(OC)cc(/C=C\C(=O)Nc3ccccc3)cc2)cc1"]

Murcko_list = [Chem.MolFromSmarts(m) for m in Murcko_list_sm]

'''

print(Murcko_list)

mcs = rdFMCS.FindMCS(Murcko_list, completeRingsOnly=True, ringMatchesRingOnly=True)

print(mcs.smartsString)

MCS_mol = Chem.MolFromSmarts(mcs.smartsString)

'''

#MCS_mol = Chem.MolFromSmarts("O=C1[*][*]C(=O)N1Cc3ncc(c2ccccc2)o3")

MCS_mol_smarts = "COC1CC(NCC(N)=O)C1"

MCS_mol = Chem.MolFromSmarts(MCS_mol_smarts)


Extras_smarts = [MCS_mol_smarts] +  [  "CCOC2CC(NC(=O)Nc1c(C)n[nH]c1C)C2" ]







AllChem.Compute2DCoords(MCS_mol)

for m in Murcko_list:

    try:
        _ = AllChem.GenerateDepictionMatching2DStructure(m, MCS_mol, allowRGroups=True)

    except:

        AllChem.Compute2DCoords(m)

_ = AllChem.GenerateDepictionMatching2DStructure(initial_hit, MCS_mol, allowRGroups=True)

draw([initial_hit], [full_TRB], output_folder + "/initial_hit")

round = 0

labels = []

cores = []

ms_copy = copy.copy(ms)

murcko_copy = copy.copy(Murcko)

for core_, sm in zip(Murcko_list, Murcko_list_sm):

    mms = []

    ms_temp_copy =[]

    murcko_temp_copy = []

    for murck,m in zip( murcko_copy, ms_copy):

        if murck == sm:

            mms.append(m)

        else:

            ms_temp_copy.append(m)
            murcko_temp_copy.append(murck)

    print( sm,  len(ms_copy),len(mms), len(ms_temp_copy))

    if len(mms) > 1:

        mcs = rdFMCS.FindMCS(mms, completeRingsOnly=True, ringMatchesRingOnly=True)

        qcore = Chem.MolFromSmarts(mcs.smartsString)

        # qcore = core_

        AllChem.Compute2DCoords(qcore)

        # img=Draw.MolsToGridImage(mms,molsPerRow=10,subImgSize=(200,200))
        # img.save(output_folder + "/R_group_decomposition_scafold_"+str(round)+" .png")

        draw(mms, [], output_folder + "/R_group_decomposition_scafold_" + str(round))

        for m in mms:
            for atom in m.GetAtoms():
                atom.SetIntProp("SourceAtomIdx", atom.GetIdx())

        rdkit.RDLogger.DisableLog('rdApp.warning')

        groups, _ = rdRGroupDecomposition.RGroupDecompose([qcore], mms, asSmiles=True, asRows=True)
        groups_mol, _ = rdRGroupDecomposition.RGroupDecompose([qcore], mms, asSmiles=False, asRows=True)

        m_core =  Chem.MolFromSmarts(  Chem.MolToSmarts( groups_mol[0]["Core"]))

        labels.append(len(mms))

        cores.append(m_core)

        try:
            _ = AllChem.GenerateDepictionMatching2DStructure(core_, MCS_mol)

        except:

            AllChem.Compute2DCoords(core_)

        df = pd.DataFrame.from_dict(groups)

        df["SMILES"] = [Chem.MolToSmiles(m) for m in mms]

        df.to_csv(output_folder + "/R_group_decomposition_" + str(round) + ".csv")

        round += 1

    ms_copy = copy.copy(ms_temp_copy)

    murcko_copy = copy.copy(murcko_temp_copy)

for m in cores:

    print(m)

    # _ = AllChem.GenerateDepictionMatching2DStructure(m,qcore)

    try:
        _ = AllChem.GenerateDepictionMatching2DStructure(m, MCS_mol, allowRGroups=True)

    except:
        AllChem.Compute2DCoords(m)




### now try to match what is left


print("ms copy",len(ms_copy))

round = 0

for sm in Extras_smarts:

    ms_copy_ = []
    ms_temp_copy = []

    qp = Chem.AdjustQueryParameters()
    qp.makeDummiesQueries = True
    qp.adjustDegree = True
    qp.adjustDegreeFlags = Chem.ADJUST_IGNOREDUMMIES

    core = Chem.MolFromSmiles(sm)

    qm = Chem.AdjustQueryProperties(core, qp)

    for m in ms_copy:

        if m.HasSubstructMatch(core):

            ms_copy_.append(m)
        else:

            ms_temp_copy.append(m)


    print( sm,  len(ms_copy), len(ms_temp_copy))

    for m in ms_copy_:
        for atom in m.GetAtoms():
            atom.SetIntProp("SourceAtomIdx", atom.GetIdx())

    rdkit.RDLogger.DisableLog('rdApp.warning')

    groups, _ = rdRGroupDecomposition.RGroupDecompose([core], ms_copy_, asSmiles=True, asRows=True)
    groups_mol, _ = rdRGroupDecomposition.RGroupDecompose([core], ms_copy_, asSmiles=False, asRows=True)

    m_core = Chem.MolFromSmarts(Chem.MolToSmarts(groups_mol[0]["Core"]))

    try:

        _ = AllChem.GenerateDepictionMatching2DStructure(m_core, MCS_mol)

    except:

        AllChem.Compute2DCoords(m_core)

    labels.append(len(ms_copy_))

    cores.append(m_core)

    df = pd.DataFrame.from_dict(groups)

    df["SMILES"] = [Chem.MolToSmiles(m) for m in ms_copy_]

    df.to_csv(output_folder + "/R_group_decomposition_" + "misc"+ str(round) + ".csv")

    round+=1

    ms_copy = copy.copy(ms_temp_copy)






left_over = copy.copy(ms_copy)

for m in left_over:

    cores.append(m)
    labels.append(1)

draw(cores, [str(l) for l in labels], output_folder + "/R_group_cores")

#draw(Murcko_list, [str(l) for l in labels], output_folder + "/Tanimoto_Murcko")
