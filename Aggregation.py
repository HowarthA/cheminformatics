import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import DataStructs
import random
import pickle
from scipy.stats import gaussian_kde
import tqdm
from matplotlib import pyplot as plt

def standardize(mol):

    clean_mol = rdMolStandardize.Cleanup(mol)
    parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
    uncharger = rdMolStandardize.Uncharger()  # annoying, but necessary as no convenience method exists
    uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
    te = rdMolStandardize.TautomerEnumerator()  # idem
    taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)

    return taut_uncharged_parent_clean_mol



Aggregations = open("/Users/alexanderhowarth/Downloads/aggregators.txt","r")

mols = []

for l in Aggregations.readlines():

    s= l.split()[0]

    m = Chem.MolFromSmiles(s)

    if m:

        m = standardize(m)

        mols.append(m)

print(len(mols))

AGG_fps =  [AllChem.GetMorganFingerprintAsBitVect(m, 3) for m in mols ]

pickle.dump(AGG_fps,open("Aggregator/AGGfps.p","wb"))

LL1 = Chem.SDMolSupplier("/Users/alexanderhowarth/Desktop/Libraries/LL1.sdf")

LL1_mols = []

LL1_inds =[ ]

i = 0

for m in LL1:

    if m:

        m = standardize(m)

        LL1_mols.append(m)

        LL1_inds.append(i)

    i+=1

LL1_fps =  [AllChem.GetMorganFingerprintAsBitVect(m, 3) for m in LL1_mols  ]

AGGregator_sims = []

LL1_sims = []

#for n in tqdm.tqdm(range(0,10000)):

 #   fp = LL1_fps[ random.randint(0,len(LL1_fps) -1) ]

  #  sim = DataStructs.BulkTanimotoSimilarity(fp, AGG_fps)

   # LL1_sims.append(max(sim))


for fp in LL1_fps:

    sim = DataStructs.BulkTanimotoSimilarity(fp, AGG_fps)

    LL1_sims.append(max(sim))

pickle.dump(LL1_sims,open("Aggregator/LL1sims.p","wb"))

for n in tqdm.tqdm( range(0, 10000)):

    id_ = random.randint(0, len(AGG_fps) - 1)

    fp = AGG_fps[id_]

    sim = DataStructs.BulkTanimotoSimilarity(fp, AGG_fps)

    sim.pop(id_)

    AGGregator_sims.append(max(sim))

pickle.dump(AGGregator_sims,open("Aggregator/Aggsims.p","wb"))



AGGregator_sims = pickle.load(open("Aggregator/Aggsims.p","rb"))
LL1_sims = pickle.load(open("Aggregator/LL1sims.p","rb"))

LL1_inds = np.array(LL1_inds)

print( LL1_inds[ np.argsort(LL1_sims)[::-1][0:10] ])

AGG_kde = gaussian_kde(AGGregator_sims)

LL1_kde = gaussian_kde(LL1_sims)

x = np.linspace(0,1,1000)

plt.plot(x, AGG_kde.pdf(x),linewidth = 3,color = "crimson")
plt.plot(x,LL1_kde.pdf(x),linewidth = 3, color = "deepskyblue")
plt.plot( x ,LL1_kde.pdf(x)/( LL1_kde.pdf(x) + AGG_kde.pdf(x)  ),linewidth = 3,color = "C4"  )
plt.show()