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



########################################################################################################################
# Do some physchem plots
########################################################################################################################


sdf = "/Users/alexanderhowarth/Desktop/Projects/DM1/TRB0050025/Pharmacophore_searches/Enamine_tsne_r3.sdf"

initial_hit = "/Users/alexanderhowarth/Desktop/Projects/DM1/TRB0050025/Pharmacophore_searches/50025_pose.sdf"

output = "/Users/alexanderhowarth/Desktop/Projects/DM1/TRB0050025/Pharmacophore_searches/"

full_TRB = "TRB0050025"

mols = [m for  m in Chem.SDMolSupplier(sdf)]

initial_hit = Chem.MolFromMolFile(initial_hit)


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


#Tan_mw ,Tan_TPSA , Tan_cLogP , Tan_fsp3 = get_props_2d_continious( Tan_Mols )


Tan_mw = [ float(Descriptors.MolWt(m)) for m in mols]

Tan_TPSA = [ float(rdMolDescriptors.CalcTPSA(m)) for m in mols]

Tan_LogP = [ float(m.GetProp("molLogP")) for m in mols]

Tan_LogD = [ float(m.GetProp("molLogD")) for m in mols]

Tan_fsp3 = [rdMolDescriptors.CalcFractionCSP3(m) for m in mols]

fig,axs = plt.subplots(ncols=3, nrows=2)

axs[0,0].axvline( Descriptors.MolWt( initial_hit ) , linewidth = 2 , color= "grey" , linestyle = ":",label = full_TRB  )

axs[0,1].axvline( 3.21 , linewidth = 2 , color= "grey" , linestyle = ":",label = full_TRB  )

axs[0,2].axvline(1.85 , linewidth = 2 , color= "grey" , linestyle = ":",label = full_TRB  )

axs[1,1].axvline( rdMolDescriptors.CalcTPSA(initial_hit), linewidth = 2 , color= "grey" , linestyle = ":",label = full_TRB  )

axs[1,0].axvline(   rdMolDescriptors.CalcFractionCSP3(initial_hit), linewidth = 2 , color= "grey" , linestyle = ":",label = full_TRB  )


def plot_props( mw ,TPSA , cLogP, cLogD, fsp3 , linestyle ,label ,alpha   ):

    x,pdf = plot_property_kde(mw)

    axs[0,0].plot(x,pdf,linewidth = 3 , color = "C0" , linestyle = linestyle , label = label ,alpha = alpha)

    axs[0,0].set_xlabel("Molecular Weight")

    #axs[0,0].set_xlim([100,300])

    ####

    #axs[0,1].set_xlim([50,160])

    ####

    x,pdf = plot_property_kde(cLogP)

    axs[0,1].plot(x,pdf,linewidth = 3 , color = "C1", linestyle = linestyle, label = label + " molLogP",alpha = alpha)

    axs[0,1].set_xlabel("molLogP")

    ####

    x,pdf = plot_property_kde(cLogD)

    axs[0,2].plot(x,pdf,linewidth = 3 , color = "C2", linestyle = linestyle, label = label+ " molLogD",alpha = alpha)

    axs[0,2].set_xlabel("molLogD")


    x,pdf = plot_property_kde(fsp3)

    axs[1,0].plot(x,pdf,linewidth = 3 , color = "C3", linestyle = linestyle, label = label,alpha = alpha)

    axs[1,0].set_xlabel("Fraction SP3")

    x,pdf = plot_property_kde(TPSA)

    axs[1,1].plot(x,pdf,linewidth = 3 , color = "C4", linestyle = linestyle, label = label,alpha = alpha)

    axs[1,1].set_xlabel("TPSA")


plot_props( Tan_mw ,Tan_TPSA , Tan_LogP ,Tan_LogD , Tan_fsp3 , "-" ,"Near Neighbour Selection" ,1 )







fig.set_size_inches(12, 8)

plt.savefig( output +"Physchem_distributions.png", format="PNG",
            bbox_inches='tight')
plt.savefig( output +  "Physchem_distributions.svg", format="SVG",
            bbox_inches='tight')

plt.close()




