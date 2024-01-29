import os
import numpy as np
from rdkit import Chem
import tqdm
import pickle
from matplotlib import pyplot as plt
from scipy.stats import spearmanr

HTRF_ids = {}

'''for m in Chem.SDMolSupplier("/Users/alexanderhowarth/Desktop/Projects/YTHDC1/docking_investigation/HTRF_with_VS.sdf"):

    try:

        ID = m.GetProp("ID")

        if ID in HTRF_ids.keys():

            HTRF_ids[ID].append(float(m.GetProp("ABSOLUTE_IC50_MEAN")))

        else:

            HTRF_ids[ID] = [float(m.GetProp("ABSOLUTE_IC50_MEAN"))]

    except:

        print("broken")

for k,v in zip(HTRF_ids.keys(),HTRF_ids.values()):

    HTRF_ids[k] = np.mean(v)
'''

docked_file = "/Users/alexanderhowarth/Desktop/Projects/DM1/TRB0050025/exvol/two_rings/comb_r3_props_filtered.sdf"

w_ = Chem.SDWriter(docked_file[:-4] + "_agg.sdf")

APFs = []
mols = []
mols_to_write = []


for m in tqdm.tqdm(Chem.SDMolSupplier(docked_file)):

    if m:

        mols.append(m)

Dict ={}

for m in tqdm.tqdm(mols):



    VENDOR_ID_ = Chem.MolToInchiKey(m)



    if VENDOR_ID_ in Dict.keys():


        score = float(m.GetProp("Score"))



        Dict[VENDOR_ID_][0].append(score)
        Dict[VENDOR_ID_][1].append(m)

    else:


        score = float(m.GetProp("Score"))

        Dict[VENDOR_ID_] = [[score],[m]]


for k in Dict.keys():

    vs = Dict[k]

    w = np.argmin(vs[0])


    m = vs[1][w]



    #m.SetProp("ABSOLUTE_IC50_MEAN" , str(HTRF_ids[k]))

    mols_to_write.append(m )

print("poses" , len(mols) , "merged" , len(mols_to_write))

scores = np.array([ float(m.GetProp("Score")) for m in mols_to_write ] )
s_ = np.argsort(scores)


for s in s_:

    w_.write(mols_to_write[s])

'''
res = []

template = []

for m in Chem.SDMolSupplier("/Users/alexanderhowarth/Desktop/Projects/YTHDC1/docking_investigation/l53551_dock_AGG.sdf"):

if m:

IC50 = -np.log10(float(m.GetProp("ABSOLUTE_IC50_MEAN")))
AFP = float(m.GetProp("median_APF"))
score = float(m.GetProp("median_Score"))

AFP_std = float(m.GetProp("stdev_APF"))
score_std = float(m.GetProp("stdev_Score"))

res.append([IC50, -1 * AFP, -1 * score, AFP_std / np.sqrt(float(m.GetProp("n_poses"))),
            score_std / np.sqrt(float(m.GetProp("n_poses")))])

if m.GetProp("VENDOR_ID") == "TRB0053551":
    template = [IC50, -1 * AFP, -1 * score]



res = np.array(res)

spr = spearmanr(res[:, 1], res[:, 0])
plt.title("Spearmans Rank " + str(round(spr.statistic, 2)))
i = 0
for r in res:
if i == 0:
plt.plot([r[0], r[0]], [r[1] + r[3], r[1] - r[3]], color="deepskyblue", alpha=0.5,
         label="Standard error in mean")
plt.plot([r[0], r[0]], [r[1] + r[3], r[1] - r[3]], color="deepskyblue", alpha=0.5)
i += 1

plt.plot(res[:, 0], res[:, 1], "o", color="deepskyblue", alpha=0.8)
plt.ylabel("-1*Median pose APF (Atomic Property Field) Score\nTRB0053551 template (more positive is better)")
plt.xlabel("Mean HTRF IC50 (-log10) (more positive is more potent)")
plt.plot(template[0], template[1], "o", color="gold", markersize=10, label="TRB0053551", alpha=0.8)
plt.legend()
plt.show()

plt.plot(res[:, 0], res[:, 2], "o", color="magenta", alpha=0.8)

i = 0
for r in res:
if i == 0:
plt.plot([r[0], r[0]], [r[2] + r[4], r[2] - r[4]], color="magenta", alpha=0.5, label="Standard error in mean")
plt.plot([r[0], r[0]], [r[2] + r[4], r[2] - r[4]], color="magenta", alpha=0.5)
i += 1
spr = spearmanr(res[:, 0], res[:, 2], )

plt.plot(template[0], template[2], "o", color="gold", markersize=10, label="TRB0053551", alpha=0.8)
plt.title("Spearmans Rank " + str(round(spr.statistic, 2)))
plt.ylabel("-1*Median Docking Score (more positive is better)")
plt.xlabel("Mean HTRF IC50 (-log10) (more positive is more potent)")
plt.legend()
plt.show()'''