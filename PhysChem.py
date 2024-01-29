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

    sims = DataStructs.BulkTanimotoSimilarity(centre_fp, fps)

    kde = gaussian_kde(sims).pdf(x)

    return x, kde

def plot_property_kde(property_list):

    kde = gaussian_kde(property_list)

    x = np.linspace(min(property_list),max(property_list),1000)

    pdf = kde.pdf(x)

    return x, pdf


def plot_props( mw ,TPSA , cLogP , fsp3 , linestyle ,label ,alpha ,axs  ):

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

    axs[1,1].plot(x,pdf,linewidth = 3 , color = "C3", linestyle = linestyle, label = label,alpha = alpha)

    axs[1,1].set_xlabel("Fraction SP3")

    axs[1,1].set_xlim([0,0.5])

def plot_molsoft_props( molCaCO2 , molPAMPA , molLogS, DILI_PredScore , linestyle ,label ,alpha ,axs   ):

    x,pdf = plot_property_kde(molCaCO2)

    axs[0,0].plot(x,pdf,linewidth = 3 , color = "C3" , linestyle = linestyle , label = label ,alpha = alpha)

    axs[0,0].set_xlabel("MolSoft CaCO2 Prediction")

    ####

    x,pdf = plot_property_kde(molPAMPA)

    axs[0,1].plot(x,pdf,linewidth = 3 , color = "C4", linestyle = linestyle, label = label,alpha = alpha)

    axs[0,1].set_xlabel("MolSoft PAMPA Prediction")

    ####

    x,pdf = plot_property_kde(molLogS)

    axs[1,0].plot(x,pdf,linewidth = 3 , color = "C5", linestyle = linestyle, label = label,alpha = alpha)

    axs[1,0].set_xlabel("MolSoft LogS Prediction")

    ####

    x,pdf = plot_property_kde(DILI_PredScore)

    axs[1,1].plot(x,pdf,linewidth = 3 , color = "C6", linestyle = linestyle, label = label,alpha = alpha)

    axs[1,1].set_xlabel("MolSoft DILI Prediction")


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


hit_smiles = "C[C@](C(F)(F)F)(C(Nc(ccc(S(c(cc1)ccc1C(NC)=O)(=O)=O)c1)c1Cl)=O)O"
full_TRB = 'TRB0052496'

code = "52496"

output_folder = "/Users/alexanderhowarth/Desktop/YTHDC1_Mcule_followup"

initial_hit =  standardize(Chem.MolFromSmiles(hit_smiles))

centre_fp = AllChem.GetMorganFingerprintAsBitVect(initial_hit, 2, 1024)

df = PandasTools.LoadSDF("/Users/alexanderhowarth/Desktop/YTHDC1_Mcule_followup/Full_selection_molsoft_tox_sol_models2.sdf")




Mols_ = [ standardize(m) for m in df.ROMol  ]

Tan_mw ,Tan_TPSA , Tan_cLogP , Tan_fsp3 = get_props_2d_continious( Mols_)


fig,axs = plt.subplots(ncols=2, nrows=2)



plot_props( Tan_mw ,Tan_TPSA , Tan_cLogP , Tan_fsp3 , "-" ,"Near Neighbour Selection" ,1 ,axs)



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

plt.savefig( output_folder +  "/Physchem_distributions.png", format="PNG",
            bbox_inches='tight')
plt.savefig( output_folder + "/Physchem_distributions.svg", format="SVG",
            bbox_inches='tight')

plt.close()


############Molsoft Property analysis



molCaCO2 = [ float(v) for v in df['molCACO2'] ]
molPAMPA = [ float(v) for v in df['molPAMPA'] ]
molLogS = [ float(v) for v in df['molLogS'] ]
DILI_PredScore = [ float(v) for v in df['DILI_PredScore'] ]

print(molCaCO2)

fig,axs = plt.subplots(ncols=2, nrows=2)
plot_molsoft_props( molCaCO2 , molPAMPA , molLogS, DILI_PredScore , "-" ,"Compound Selection" ,1,axs )



axs[0,0].axvline( -5.29 , linewidth = 2 , color= "grey" , linestyle = ":",label = full_TRB  )

axs[0,1].axvline(-5.37 , linewidth = 2 , color= "grey" , linestyle = ":",label = full_TRB  )

axs[1,0].axvline(-3.38 , linewidth = 2 , color= "grey" , linestyle = ":",label = full_TRB  )

axs[1,1].axvline(  0.51 , linewidth = 2 , color= "grey" , linestyle = ":",label = full_TRB  )


axs[0, 0].legend()
axs[0, 1].legend()
axs[1, 0].legend()
axs[1, 1].legend()

fig.set_size_inches(12, 8)

plt.savefig( output_folder +  "/Molsoft_Physchem_distributions.png", format="PNG",
            bbox_inches='tight')
plt.savefig( output_folder + "/Molsoft_Physchem_distributions.svg", format="SVG",
            bbox_inches='tight')

plt.close()