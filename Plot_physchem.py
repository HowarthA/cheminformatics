import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor
import random
rdDepictor.SetPreferCoordGen(True)
import numpy as np
from rdkit.Chem import AllChem
from rdkit import RDLogger
from matplotlib import pyplot as plt
from rdkit.Chem.MolStandardize import rdMolStandardize
import tqdm
from scipy.stats import gaussian_kde
from rdkit.Chem import Descriptors
from rdkit.Chem import Crippen
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.SimDivFilters import rdSimDivPickers
from rdkit.Chem import PandasTools

lp_minmax = rdSimDivPickers.MaxMinPicker()
lp_leader = rdSimDivPickers.LeaderPicker()

def standardize(mol):

    clean_mol = rdMolStandardize.Cleanup(mol)
    parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
    uncharger = rdMolStandardize.Uncharger()  # annoying, but necessary as no convenience method exists
    uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
    te = rdMolStandardize.TautomerEnumerator()  # idem
    taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)

    return taut_uncharged_parent_clean_mol

def remove_duplicates_by_inchi_key(smiles_list):
    uncharger = rdMolStandardize.Uncharger()
    te = rdMolStandardize.TautomerEnumerator()

    Mols = []
    inchis = []

    for s in smiles_list:

        m = Chem.MolFromSmiles(s)

        if m:
            m = standardize(m)

            clean_mol = rdMolStandardize.Cleanup(m)
            parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
            uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
            m = te.Canonicalize(uncharged_parent_clean_mol)

            i = Chem.MolToInchiKey(m)

            if i not in inchis:
                Mols.append(m)

    return Mols

def compare_different_dataset(fps1 , fps2):

    x = np.linspace(0,1,1000)

    maxes = np.zeros(len(fps1))

    i = 0

    for fp in tqdm.tqdm(fps1):

        sims = DataStructs.BulkTanimotoSimilarity(fp, fps2)

        maxes[i] = max(sims)

        i +=1

    kde = gaussian_kde(maxes).pdf(x)

    return x, kde

def compare_different_dataset_images(fps1 , fps2,mols_1,mols_2):

    x = np.linspace(0,1,1000)

    maxes = np.zeros(len(fps1))

    i = 0

    for fp in tqdm.tqdm(fps1):

        sims = DataStructs.BulkTanimotoSimilarity(fp, fps2)

        maxes[i] = max(sims)

        if max(sims) > 0.99:
            img = Draw.MolsToGridImage([mols_1[i], mols_2[np.argmax(np.array(sims))]])

            img.save("images/" + "Duplicates" +str(random.randint(1000,9999))+ ".png")

        i +=1

    kde = gaussian_kde(maxes).pdf(x)

    return x, kde

def compare_same_dataset(centre_fp,fps ):

    x = np.linspace(0,1,1000)

    i = 0

    sims = DataStructs.BulkTanimotoSimilarity(centre_fp, fps)[0]

    kde = gaussian_kde(sims).pdf(x)

    return x, kde

output_folder = "5411"

hit_smiles = "COc(cc(/C=C(\C(Nc1ccccc1)=O)/C#N)cc1)c1OCc(cc1)ccc1C(O)=O"

initial_hit =  standardize(Chem.MolFromSmiles(hit_smiles))

centre_fp = AllChem.GetMorganFingerprintAsBitVect(initial_hit, 2, 1024)

df = PandasTools.LoadSDF("/Users/alexanderhowarth/Desktop/EML4-quotes/labeled_full_quote.sdf")

#Mols = [standardize(m) for m in tqdm.tqdm(Chem.SDMolSupplier("/Users/alexanderhowarth/Desktop/EML4-quotes/labeled_full_quote.sdf"))]


fps = [ AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024) for m in tqdm.tqdm(df.ROMol)  ]

#compare_different_dataset_images(Sellec_fps , FDA_fps,Selleck_Mols,FDA_Mols)

x, pdf = compare_same_dataset(centre_fp,fps)

plt.plot(x,pdf,label = "Acrylamide Fragments NN Distribution",linewidth = 3)

plt.plot(x,pdf,label = "Acrylamide Fragments similarity to 10k-LL",linewidth = 3)

plt.xlabel("Tanimoto Similarity")
plt.ylabel("Density")

plt.legend()

plt.savefig(output_folder+ "/NN_distributions.png",format = "PNG",bbox_inches='tight')
plt.savefig(output_folder+"/NN_distributions.svg",format = "SVG",bbox_inches='tight')

plt.close()

########################################################################################################################
# Do some physchem plots
########################################################################################################################

def plot_property_kde(property_list):

    kde = gaussian_kde(property_list)

    x = np.linspace(min(property_list),max(property_list),1000)

    pdf = kde.pdf(x)

    return x, pdf


def get_props_2d_continious(m):
    return [Descriptors.MolWt(m), rdMolDescriptors.CalcTPSA(m), Crippen.MolLogP(m),
            rdMolDescriptors.CalcFractionCSP3(m)]

#df_2 = pd.DataFrame([get_props_2d_continious(m) for m in Mols], columns=['mw', 'TPSA', 'cLogP', 'Fraction SP3'])


mw_ ,TPSA_ , cLogP , fsp3 = [get_props_2d_continious(m) for m in df.ROMol]


df['MW'] = mw_
df['TPSA'] = TPSA_
df['cLogP'] = cLogP
df['fsp3'] = fsp3

print(df)

quit()


fig,axs = plt.subplots(ncols=2, nrows=2)

x,pdf = plot_property_kde(df['mw'])

axs[0,0].plot(x,pdf,linewidth = 3 , color = "C0")

axs[0,0].set_xlabel("Molecular Weight")

axs[0,0].set_xlim([0,1000])

axs[0,0].legend()

####

x,pdf = plot_property_kde(df['TPSA'])

axs[0,1].plot(x,pdf,linewidth = 3 , color = "C1")

axs[0,1].set_xlim([0,500])
axs[0,1].set_xlabel("TPSA")
axs[0,1].legend()

####

x,pdf = plot_property_kde(df['cLogP'])

axs[1,0].plot(x,pdf,linewidth = 3 , color = "C2")



axs[1,0].set_xlabel("cLogP")
axs[1,0].set_xlim([-10,10])
axs[1,0].legend()

####

x,pdf = plot_property_kde(df['Fraction SP3'])

axs[1,1].plot(x,pdf,linewidth = 3 , color = "C2",alpha = 0.5)

axs[1,1].set_xlabel("Fraction SP3")

axs[1,1].legend()

fig.set_size_inches(12, 8)

plt.savefig( output_folder +  "/Physchem_distributions.png", format="PNG",
            bbox_inches='tight')
plt.savefig( output_folder + "/Physchem_distributions.svg", format="SVG",
            bbox_inches='tight')

plt.close()

