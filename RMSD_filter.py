from rdkit import Chem
import numpy as np
from rdkit.Chem import rdmolops
from sklearn.metrics.pairwise import euclidean_distances

ref = Chem.MolFromMolFile("/Users/alexanderhowarth/Desktop/Projects/DM1/TRB0050025/Pharmacophore_searches/50025_pose.sdf")

mols =[ m for m in Chem.SDMolSupplier("/Users/alexanderhowarth/Desktop/Projects/DM1/TRB0050025/exvol/two_rings/comb_r2_props_thresh_filter_confs_ans_agg.sdf") if m ]

writer = Chem.SDWriter("/Users/alexanderhowarth/Desktop/Projects/DM1/TRB0050025/exvol/two_rings/comb_r2_props_thresh_filter_confs_ans_agg_thresh.sdf")

ref_pos = ref.GetConformer().GetPositions()

print(len(ref_pos))

l =0

for m in mols:

    pos_m = m.GetConformer().GetPositions()

    '''print(np.shape(ref_pos))
    print(np.shape(pos_m))

    print(np.shape(euclidean_distances(pos_m , ref_pos)))

    print(np.shape(np.min(euclidean_distances(pos_m , ref_pos),axis = 0)))
    print(np.max(np.min(euclidean_distances(pos_m , ref_pos),axis = 0)))
    quit()'''

    if rdmolops.GetFormalCharge(m) < 0:
        continue

    if np.max(np.min(euclidean_distances(pos_m , ref_pos),axis = 0)) < 2:

        writer.write(m)
        l+=1


print(l)