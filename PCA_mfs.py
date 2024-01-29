from rdkit import Chem
from rdkit.Chem import AllChem
from matplotlib import pyplot as plt
from rdkit import DataStructs
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering


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


mols = [m for m in Chem.SDMolSupplier(
    "/Users/alexanderhowarth/Desktop/Projects/DM1/TRB0050025/Pharmacophore_searches/Enamine.sdf")]

print(len(mols))

fps = np.array([AllChem.GetMorganFingerprintAsBitVect(m, 3) for m in mols])

'''sims_0 = np.array([ DataStructs.BulkTanimotoSimilarity( fps[0] , fps )  ][0])

print(sims_0)

min_sim = np.argmin(sims_0)

sims_1 = np.array([ DataStructs.BulkTanimotoSimilarity( fps[min_sim] , fps )  ][0])


plt.plot(sims_0,sims_1,"o",alpha = 0.5)

plt.show()'''

'''X_embedded = TSNE(n_components=2, learning_rate='auto',
              init='random', perplexity=1000).fit_transform(fps)
'''

pca_2 = PCA(n_components=2)

X_embedded = pca_2.fit_transform(fps)

print(X_embedded.shape)

n_clusters = 6
clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(X_embedded)

print(clustering.labels_)

for m, c in zip(mols, clustering.labels_):
    m.SetProp("cluster", str(c))

mols_to_draw = []

for n in range(n_clusters):
    w = np.where(clustering.labels_ == n)

    print(w[0][0])

for x, c in zip(X_embedded, clustering.labels_):
    plt.plot(x[0], x[1], "o", alpha=0.2, color="C" + str(c % 10))

plt.show()