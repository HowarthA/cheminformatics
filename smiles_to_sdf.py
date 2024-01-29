from rdkit import Chem
import os
f = "/Users/alexanderhowarth/Desktop/Projects/AR/Rob_AR_designs.smi"
writer = Chem.SDWriter(f[:-4] + ".sdf")
f = open(f,"r")




for l in f.readlines():

    m = Chem.MolFromSmiles(l)

    if m:

        writer.write(m)