from rdkit import Chem
from rdkit.Chem import AllChem
from matplotlib import pyplot as plt
from rdkit import DataStructs
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from rdkit.Chem.Scaffolds import MurckoScaffold
import tqdm
from rdkit import SimDivFilters

def draw(mols, labels, file):

    from rdkit.Chem.Draw import rdMolDraw2D

    n_per_row = min(4 , len(mols))

    scale = 300

    n_rows = int(len(mols) / n_per_row) + (len(mols) % n_per_row > 0)

    d2d = rdMolDraw2D.MolDraw2DSVG(n_per_row * scale, n_rows * scale, scale, scale)

    d2d.DrawMolecules(list(mols), legends=labels)

    pic = open(file + ".svg", "w+")

    d2d.FinishDrawing()

    pic.write(str(d2d.GetDrawingText()))

    pic.close()

import pickle
#
#lib_fps = [ AllChem.GetMorganFingerprintAsBitVect(m,3) for m in Chem.SDMolSupplier("/Users/alexanderhowarth/Desktop/Libraries/Full_screening_library.sdf") if m]

#pickle.dump(lib_fps,open("lib_fps.p","wb"))

lib_fps = pickle.load(open("lib_fps.p","rb"))

mols  = []

'''for m in Chem.SDMolSupplier("/Users/alexanderhowarth/Desktop/Projects/DM1/TRB0050025/exvol/two_rings/comb_r3_props_agg_filtered.sdf"):

    if m:

        fp = AllChem.GetMorganFingerprintAsBitVect(m,3)

        sims = DataStructs.BulkTanimotoSimilarity(fp,lib_fps)

        if max(sims) < 0.8:

            mols.append(m)

        else:

            print(Chem.MolToSmiles(m))'''


mols = [m for  m in Chem.SDMolSupplier("/Users/alexanderhowarth/Desktop/Projects/DM1/TRB0050025/exvol/two_rings/comb_r2_props_thresh_filter.sdf")]


print(len(mols))

fps = [ AllChem.GetMorganFingerprintAsBitVect(m,3) for m in mols]

mols = np.array(mols)

#### Do Murko scaffold pick

mur = {}

scaffolds = []
mol_scaffold_number = []

counter = 0

for m in mols:

    s = MurckoScaffold.MurckoScaffoldSmilesFromSmiles(Chem.MolToSmiles(m))
    ms =  MurckoScaffold.GetScaffoldForMol(m)

    if s not in mur.keys():

        mur[s] = counter
        mol_scaffold_number.append(mur[s])
        counter+=1
        scaffolds.append(ms)

    else:

        mol_scaffold_number.append(mur[s])

mol_scaffold_number = np.array(mol_scaffold_number)

writer = Chem.SDWriter("/Users/alexanderhowarth/Desktop/Projects/DM1/TRB0050025/Pharmacophore_searches/Enamine_scaffold_r2.sdf")

for id_ in range(0,len(mur.keys())):

    mols_with_scffold = mols[ mol_scaffold_number == id_]

    mols_with_scffold_fps = [ fps[i] for i in np.where(mol_scaffold_number == id_)[0] ]

    av_sim = []

    for  i ,m in enumerate(mols_with_scffold):

        sims = np.array([ DataStructs.BulkTanimotoSimilarity( mols_with_scffold_fps[i] , mols_with_scffold_fps)  ][0])
        av_sim.append(np.sum(sims))

    max_sim = np.argmax(av_sim)

    writer.write(mols_with_scffold[max_sim])

##Do fps pick




dims = np.arange(1,25,2)

res = []

for n_comp in tqdm.tqdm(dims):
    pca = PCA(n_components=n_comp)
    crds = pca.fit_transform(fps)
    var = np.sum(pca.explained_variance_ratio_)
    res.append(var)

plt.plot(dims, res)
plt.show()


pca_2= PCA(n_components=15)

X_embedded = pca_2.fit_transform(np.array(fps))

X_embedded = TSNE(n_components=2).fit_transform(X_embedded)

print("ratio" , np.sum(pca_2.explained_variance_ratio_))

print(X_embedded.shape)

from scipy.spatial.distance import cdist

from sklearn.cluster import KMeans

res =[]

n_cl = np.arange(1,20)

for n in n_cl:

    kmeans = KMeans(n_clusters=n)
    kmeans.fit(X_embedded)
    res.append(np.average(np.min(cdist(X_embedded, kmeans.cluster_centers_, 'euclidean'), axis=1)))

plt.plot(n_cl, res)
plt.title('elbow curve')
plt.show()



from sklearn.cluster import KMeans
clustering = KMeans(n_clusters=12)
clustering.fit(X_embedded)


#clustering = AgglomerativeClustering(n_clusters=20).fit(X_embedded)

print(len(clustering.labels_))

writer = Chem.SDWriter("/Users/alexanderhowarth/Desktop/Projects/DM1/TRB0050025/Pharmacophore_searches/Enamine_tsne_r2.sdf")

for m ,c  in zip(mols,clustering.labels_ ):

    m.SetProp("cluster" , str(c))

mols_to_draw = []

cluster_score = []

c_l = np.array( [c for c in clustering.labels_  ]  )

for n in set(clustering.labels_):

    w = np.where(clustering.labels_ == n)

    #find the centroid:

    av_sim = []

    for  i ,m in enumerate(mols[w[0]]):

        sims = np.array([ DataStructs.BulkTanimotoSimilarity( fps[w[0][i]] , [fps[j] for j in  w[0]] )  ][0])
        av_sim.append(np.sum(sims))

    max_sim = np.argmax(av_sim)

    mols_to_draw.append(mols[w[0][max_sim]])

    writer.write(mols[w[0][max_sim]])

for m in mols_to_draw:

    AllChem.Compute2DCoords(m)

'''min_s = min(scores)
scores = [ cs - min_s for cs in scores ]
max_s = max(scores)
scores= [ cs/max_s for cs in scores]'''

import matplotlib
cmap = matplotlib.cm.get_cmap('viridis')

c = 0

for x ,l    in zip(X_embedded,clustering.labels_ ):

    plt.plot( x[0] , x[1], "o", alpha =0.5 , color = "C" + str(l%10) )

    c+=1

plt.show()

draw(mols_to_draw,[ str(i) for i in range(len(mols_to_draw)) ],"/Users/alexanderhowarth/Desktop/Projects/DM1/TRB0050025/Pharmacophore_searches/cluster_mols_r2")