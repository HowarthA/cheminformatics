from rdkit import Chem
from rdkit.Chem import AllChem

m = Chem.MolFromMolFile("/Users/alexanderhowarth/Desktop/Projects/AR/EPI-7386_core_R_confs.sdf")

R1 = Chem.MolFromMolFile("/Users/alexanderhowarth/Desktop/Projects/AR/R1.sdf")



print(Chem.MolToSmiles(m))

#####
rxn_1 = AllChem.ReactionFromSmarts('[O:1]CCCl>>CC(Cl)C[O:1]')

product = rxn_1.RunReactants((R1, m))[0][0]

print(product)
print(Chem.MolToSmiles(product))

quit()

#####


print(Chem.MolToSmiles(m))


for a1 in m.GetAtoms():

    #find which atom is oxygen bonded to I

    if a1.GetSymbol() == "O":

        for a2 in a1.GetNeighbors():

            if a2.GetSymbol() == "I":

                print(a1.GetIdx())

                #form bond between this atom and atom 1 in the R1

