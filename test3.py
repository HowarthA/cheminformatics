from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Crippen
from rdkit.Chem import rdMolDescriptors

from sklearn.preprocessing import StandardScaler

import numpy as np

from matplotlib import pyplot as plt

def descriptor(m):

    c = 0
    mw = Descriptors.MolWt(m)

    TPSA =  rdMolDescriptors.CalcTPSA(m)

    logp = Crippen.MolLogP(m)
    fsp3 = rdMolDescriptors.CalcFractionCSP3(m)

    return [mw,TPSA,logp,fsp3]

mols = []
libs =[]

lib ={}

for m in Chem.SDMolSupplier("/Users/alexanderhowarth/Desktop/Libraries/TL1_2d.sdf"):

    try:
        lib[m.GetProp("ID")] = m.GetProp("STRUCTURE_COMMENT")
    except:

        None

for m in Chem.SDMolSupplier("/Users/alexanderhowarth/Desktop/Libraries/TL1_3d_sample.sdf"):

    if m:

        try:

            libs.append(lib[m.GetProp("ID")])

            mols.append( m )

        except:

            None


print(len(mols))

mols = np.array(mols)
libs = np.array(libs)

names = sorted(list(set( libs)))


descs = np.load("/Users/alexanderhowarth/PycharmProjects/DiffSBDD_swap/TL1_reps_100_steps.npy")



from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde



descs = StandardScaler().fit_transform(descs)

'''pcs = PCA(n_components=100)

descs = pcs.fit_transform(descs)
'''
ts = TSNE(n_components=2)

descs = ts.fit_transform(descs)



print(np.shape(descs))

'''for p in range(4):

    print(mols[:,p])

    pMin = np.min(mols[:,p ] )
    pMax = np.max(mols[:,p] )
    
    x = np.linspace(pMin,pMax,100)

    for c, n in enumerate(names):


        w = np.where(libs  == n)

        print(w)

        kde = gaussian_kde( mols[w,p] )

        plt.plot( x ,kde.pdf(x), color = "C" + str(c),alpha = 1)

    plt.legend()
    plt.show()'''

for c, n in enumerate(names):

    w = np.where(libs == n)

    print(len(w[0]))

    plt.plot(descs[w,0],descs[w,1],"o", color="C" + str(c), alpha=0.3)

plt.show()