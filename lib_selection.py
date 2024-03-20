from rdkit import Chem
import tqdm
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np
from matplotlib import pyplot as plt
from rdkit.Chem.MolStandardize import rdMolStandardize
#input mols
from rdkit.Chem import Descriptors

lib_mols = "/Users/alexanderhowarth/Desktop/Projects/DM1/DM1_Screening_Library_design/Enamine_Fragment_Collection_259380cmpds_20230413.sdf"

uncharger = rdMolStandardize.Uncharger()


mw_filter = 200

def standardize(mol):

    clean_mol = rdMolStandardize.Cleanup(mol)
    parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
    # annoying, but necessary as no convenience method exists
    uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)

    return uncharged_parent_clean_mol


lib_mols_ = []
lib_fps_ = []

for m_ in tqdm.tqdm( Chem.SDMolSupplier(lib_mols)):

    if m_:

        m = standardize(m_)

        if Descriptors.MolWt(m) < mw_filter:

            lib_mols_.append(m)
            lib_fps_.append(Chem.AllChem.GetMorganFingerprintAsBitVect(m,3))

#input bases

C_G_bases = [m for m in Chem.SDMolSupplier("/Users/alexanderhowarth/Desktop/Projects/DM1/DM1_Screening_Library_design/CG_bases.sdf")]

#calculate tversky similarity to bases - allowing for extension from second fragment

C_sim = DataStructs.BulkTverskySimilarity(Chem.AllChem.GetMorganFingerprintAsBitVect(C_G_bases[0],3) , lib_fps_ , a=1,b=0  )

G_sim = DataStructs.BulkTverskySimilarity(Chem.AllChem.GetMorganFingerprintAsBitVect(C_G_bases[1],3) , lib_fps_ , a=1,b=0  )

#find max similarty

max_sim = np.max([ C_sim,G_sim ], axis = 0)


n =[ ]
for t in  [0.1,0.2,0.3,0.4,0.5] :

    n.append(np.sum( max_sim > t ))

plt.plot([0.1,0.2,0.3,0.4,0.5] , n  )

plt.show()

tvesrky_filtered = Chem.SDWriter("/Users/alexanderhowarth/Desktop/Projects/DM1/DM1_Screening_Library_design/tversky_Enamine_0.3_200.sdf")

for b , m in zip(max_sim , lib_mols_):

    if b > 0.3:

        tvesrky_filtered.write(m)