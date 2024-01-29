from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from rdkit import DataStructs
from rdkit import SimDivFilters

NNs = [m for m in Chem.SDMolSupplier("/Users/alexanderhowarth/Desktop/Projects/YTHDC1/TRB0031642_NNs/NNS_instock.sdf")]

print(len(NNs))

hit_mol = [m for m in Chem.SDMolSupplier("/Users/alexanderhowarth/Desktop/Projects/YTHDC1/TRB0031642_NNs/TRB0031642.sdf")][0]


'''fs = [
"/Users/alexanderhowarth/Desktop/Projects/YTHDC1/TRB0031642_NNs/SS_5_REAL.sdf",
"/Users/alexanderhowarth/Desktop/Projects/YTHDC1/TRB0031642_NNs/SS_4_REAL.sdf",
"/Users/alexanderhowarth/Desktop/Projects/YTHDC1/TRB0031642_NNs/SS_3_REAL.sdf",
"/Users/alexanderhowarth/Desktop/Projects/YTHDC1/TRB0031642_NNs/SS_2_REAL.sdf",
"/Users/alexanderhowarth/Desktop/Projects/YTHDC1/TRB0031642_NNs/SS_1_REAL.sdf",
"/Users/alexanderhowarth/Desktop/Projects/YTHDC1/TRB0031642_NNs/NNs_REAL.sdf",

]


NNs = []

l = ['a','b','c','d','e','f']
c=0
for f in fs:

    for m in Chem.SDMolSupplier(f):
        m.SetProp("set", l[c] )
        NNs.append(m)

    c+=1'''


NNs = [m for m in Chem.SDMolSupplier("/Users/alexanderhowarth/Desktop/Projects/YTHDC1/TRB0031642_NNs/NNs_REAL_all_props.sdf") if m]

print(NNs)


NNs_fps = [ AllChem.GetMorganFingerprintAsBitVect(m, 3) for m in NNs ]

hit_fp = AllChem.GetMorganFingerprintAsBitVect(hit_mol, 3)

writer = Chem.SDWriter("/Users/alexanderhowarth/Desktop/Projects/YTHDC1/TRB0031642_NNs/NNS_REAL_selection_2.sdf")

sims = np.array(DataStructs.BulkTanimotoSimilarity(hit_fp, NNs_fps))

sets = np.array([ m.GetProp("set") for m in NNs ])


closest_mols = []

NNs  = np.array(NNs)

l = ['a','b','c','d','e','f']

for s in l:

    w = np.where(sets == s)[0]
    print(w)
    w_2 = np.where(sims[w] > 0.4)[0]

    fps_NN = [ NNs_fps[i] for i in w ]

    fps_NN = [ fps_NN[i] for i in w_2 ]



    fps_NN = list(fps_NN)

    mmp = SimDivFilters.MaxMinPicker()

    n_pick = min([10, len(fps_NN)])

    bv_ids = mmp.LazyBitVectorPick(fps_NN, len(fps_NN), n_pick)

    closest_mols.extend([ NNs[ w[w_2[b]] ] for b in bv_ids ])


print(len(closest_mols))

for m in closest_mols:

    writer.write(m)

quit()

