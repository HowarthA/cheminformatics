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
import copy

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

    sims = DataStructs.BulkTanimotoSimilarity(centre_fp, fps)

    kde = gaussian_kde(sims).pdf(x)

    return x, kde




#5411
#hit_smiles = "COc(cc(/C=C(\C(Nc1ccccc1)=O)/C#N)cc1)c1OCc(cc1)ccc1C(O)=O"

#4618
#hit_smiles = "CCS(C(N1C(/C2=C/c(cc3)cc(OC)c3OC(c3ccccc3)=O)=N)=NSC1=NC2=O)(=O)=O"

#1264

#hit_smiles = "Cc1nc(O[C@@H](C(c2ccccc2)(c2ccccc2)OC)C(O)=O)nc(C)c1"


#4862
#hit_smiles= "Cc(c1cccnc11)cc(C(C(F)(F)F)(C(OC)=O)O)c1O"

#3486

#hit_smiles = "Cc1cccc(-c2nc(CCCC3)c3c(SCC(O)=O)n2)c1"

#176

#hit_smiles = "Fc1ccc(cc1)-c1cnc(CN2C(=O)NC3(CCSC3)C2=O)o1"
#full_TRB = 'TRB0023760'



hit_smiles = "CCOC1CC(N(C)CC(=O)Nc2c(C)n[nH]c2C)C11CCCCC1"
full_TRB = 'TRB0023566'

initial_hit =  standardize(Chem.MolFromSmiles(hit_smiles))

centre_fp = AllChem.GetMorganFingerprintAsBitVect(initial_hit, 2, 1024)

output_folder = "/Users/alexanderhowarth/Desktop/Projects/EML4-ALK/EML4-ALK_TL/TRB0023566/Enamine_selections"

#df = pd.read_csv("/Users/alexanderhowarth/Desktop/Projects/EML4-ALK/EML4-ALK_TL/TRB0023566/Enamine_selections/Z1263185411_RDB_3573analogues.sdf")



'''

Mols_ = []

for i , r in df.iterrows():

    if r['Series'] == full_TRB:

        Mols_.append( standardize( Chem.MolFromSmiles(r['SMILES']) )  )

'''

Mols_ = [ m for m in Chem.SDMolSupplier("/Users/alexanderhowarth/Desktop/Projects/EML4-ALK/EML4-ALK_TL/TRB0023566/Enamine_selections/Z1263185411_RDB_3573analogues.sdf") ]

Mols_ = [ Mols_[i] for i in range(0,min(len(Mols_ ), 50)) ]


'''
import json

series_ = [ s for s in df.series  ]

BOAW_Mols = []
Tan_Mols = []

for s,m in zip(series_,Mols_):

    if code in s:

        if "BOAW" in s:

            BOAW_Mols.append(m)

        else:

            Tan_Mols.append(m)

'''


Tan_Mols = copy.copy(Mols_)

#Mols = [standardize(m) for m in tqdm.tqdm(Chem.SDMolSupplier("/Users/alexanderhowarth/Desktop/EML4-quotes/labeled_full_quote.sdf"))]


Tan_fps = [ AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024) for m in tqdm.tqdm(Tan_Mols)  ]

print(Tan_fps)

#BOAW_fps = [ AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024) for m in tqdm.tqdm(BOAW_Mols)  ]

#compare_different_dataset_images(Sellec_fps , FDA_fps,Selleck_Mols,FDA_Mols)

print(Tan_fps)

print(centre_fp)

x, pdf = compare_same_dataset(centre_fp,Tan_fps)

plt.plot(x,pdf,label = "Tanimoto Selection",linewidth = 3,color = "C0",alpha = 0.6)


#x, pdf = compare_same_dataset(centre_fp,BOAW_fps)

#plt.plot(x,pdf,label = "Fuzzy Pharmacophore Selection",linewidth = 3,color = "C0",linestyle = "--")

plt.xlabel("Tanimoto Similarity")

plt.ylabel("Density")

plt.legend()

plt.savefig(output_folder+ "/" + full_TRB +"_NN_distributions.png",format = "PNG",bbox_inches='tight')
plt.savefig(output_folder+"/" + full_TRB +"_NN_distributions.svg",format = "SVG",bbox_inches='tight')

plt.close()

########################################################################################################################
# Do some physchem plots
########################################################################################################################

def plot_property_kde(property_list):

    kde = gaussian_kde(property_list)

    x = np.linspace(min(property_list),max(property_list),1000)

    pdf = kde.pdf(x)

    return x, pdf


def get_props_2d_continious(mols):


    mw =[ ]

    TPSA = []

    logp = []
    fsp3 = []

    for m in mols:

        mw.append(Descriptors.MolWt(m))

        TPSA.append( rdMolDescriptors.CalcTPSA(m))

        logp.append(Crippen.MolLogP(m))
        fsp3.append(rdMolDescriptors.CalcFractionCSP3(m))

    return mw,TPSA,logp,fsp3

#df_2 = pd.DataFrame([get_props_2d_continious(m) for m in Mols], columns=['mw', 'TPSA', 'cLogP', 'Fraction SP3'])


Tan_mw ,Tan_TPSA , Tan_cLogP , Tan_fsp3 = get_props_2d_continious( Tan_Mols )



fig,axs = plt.subplots(ncols=2, nrows=2)


def plot_props( mw ,TPSA , cLogP , fsp3 , linestyle ,label ,alpha   ):

    x,pdf = plot_property_kde(mw)

    axs[0,0].plot(x,pdf,linewidth = 3 , color = "C0" , linestyle = linestyle , label = label ,alpha = alpha)

    axs[0,0].set_xlabel("Molecular Weight")

    #axs[0,0].set_xlim([100,300])

    ####

    x,pdf = plot_property_kde(TPSA)

    axs[0,1].plot(x,pdf,linewidth = 3 , color = "C1", linestyle = linestyle, label = label,alpha = alpha)


    axs[0,1].set_xlabel("TPSA")

    #axs[0,1].set_xlim([50,160])

    ####

    x,pdf = plot_property_kde(cLogP)

    axs[1,0].plot(x,pdf,linewidth = 3 , color = "C2", linestyle = linestyle, label = label,alpha = alpha)

    axs[1,0].set_xlabel("cLogP")

    axs[1,0].set_xlim([1,7])

    ####

    x,pdf = plot_property_kde(fsp3)

    axs[1,1].plot(x,pdf,linewidth = 3 , color = "C2", linestyle = linestyle, label = label,alpha = alpha)

    axs[1,1].set_xlabel("Fraction SP3")

    axs[1,1].set_xlim()


plot_props( Tan_mw ,Tan_TPSA , Tan_cLogP , Tan_fsp3 , "-" ,"Near Neighbour Selection" ,1 )



axs[0,0].axvline( Descriptors.MolWt( initial_hit ) , linewidth = 2 , color= "grey" , linestyle = ":",label = full_TRB  )

axs[0,1].axvline( rdMolDescriptors.CalcTPSA(initial_hit), linewidth = 2 , color= "grey" , linestyle = ":",label = full_TRB  )

axs[1,0].axvline( Crippen.MolLogP(initial_hit), linewidth = 2 , color= "grey" , linestyle = ":",label = full_TRB  )
axs[1,1].axvline(   rdMolDescriptors.CalcFractionCSP3(initial_hit), linewidth = 2 , color= "grey" , linestyle = ":",label = full_TRB  )

axs[0,0].axvline(500 , linewidth = 2 , color= "black" , linestyle = "-",label = "<x Ideal oral drug-like range" )

axs[0,1].axvline( 140, linewidth = 2 , color= "black" , linestyle = "-",label = "<x Ideal oral drug-like range"   )

axs[1,0].axvline( 5, linewidth = 2 , color= "black" , linestyle = "-",label = "<x Ideal oral drug-like range"  )


axs[0, 0].legend()
axs[0, 1].legend()
axs[1, 0].legend()
axs[1, 1].legend()

fig.set_size_inches(12, 8)

plt.savefig( output_folder +  "/" + full_TRB +"_Physchem_distributions.png", format="PNG",
            bbox_inches='tight')
plt.savefig( output_folder + "/" + full_TRB +"_Physchem_distributions.svg", format="SVG",
            bbox_inches='tight')

plt.close()




