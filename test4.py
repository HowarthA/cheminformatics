import os

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDistGeom


original = Chem.MolFromMolFile("/Users/alexanderhowarth/Desktop/Projects/AR/EPI/EPI-7386.sdf")

original_fp = AllChem.GetMorganFingerprintAsBitVect(original,3)


m = Chem.MolFromMolFile("/Users/alexanderhowarth/Desktop/Projects/AR/EPI/EPI-7386_core_R_confs.sdf")

R1 = Chem.MolFromMolFile("/Users/alexanderhowarth/Desktop/Projects/AR/EPI/R1.sdf")
R2 = Chem.MolFromMolFile("/Users/alexanderhowarth/Desktop/Projects/AR/EPI/R2.sdf")


#####

rxn_1 = AllChem.ReactionFromSmarts('[#6:1][#53].[#8:3][#53]>>[#6:1][#8:3]')

rxn_2 = AllChem.ReactionFromSmarts('[#6:1][#53].[#8:2][#9]>>[#6:1][#8:2]')

product = rxn_1.RunReactants((R1, m))[0][0]

writer  = Chem.SDWriter("/Users/alexanderhowarth/Desktop/Projects/AR/EPI/swaps_full.sdf")

param = rdDistGeom.ETKDGv3()

for f in sorted(os.listdir("/Users/alexanderhowarth/Desktop/Projects/AR/EPI/EPI_swap2/" )):

    if f.endswith(".sdf"):

        for m in Chem.SDMolSupplier("/Users/alexanderhowarth/Desktop/Projects/AR/EPI/EPI_swap2/" + f):

            try:
                product_1 = rxn_1.RunReactants((R1, m))[0][0]

                product_2 = rxn_2.RunReactants((R2, product_1))[0][0]

                Chem.SanitizeMol(product_2)
                print(Chem.MolToSmiles(product_2))
                product_2 = AllChem.AddHs(product_2)

                cids = rdDistGeom.EmbedMultipleConfs(product_2, 1, param)

                AllChem.MMFFOptimizeMoleculeConfs(product_2, numThreads=0, mmffVariant='MMFF94s')

                product_2 = AllChem.RemoveAllHs(product_2)

                p_fp = AllChem.GetMorganFingerprintAsBitVect(product_2, 3)
                sim = AllChem.DataStructs.TanimotoSimilarity(original_fp,p_fp)

                print(sim)
                product_2.SetProp("swap" , str(f))

                product_2.SetProp("unstable_cresset" , str(m.GetProp("Unstable")))
                product_2.SetProp("swap_score" , str(m.GetProp("Score")))
                product_2.SetProp("Tanimoto_to_EPI-7386" , str(round(sim,3)))
                writer.write(product_2)
            except:

                print("borken")


print(product)
print(Chem.MolToSmiles(product))
