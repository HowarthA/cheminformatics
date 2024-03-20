from rdkit import Chem
import os
folder_ = "/Users/alexanderhowarth/Desktop/Projects/YTHDC1/Y_swap/swaps/"

writer =  Chem.SDWriter(folder_ +  "combined.sdf")

for f in os.listdir(folder_):

    if f.endswith(".sdf") and (f != "combined.sdf"):

        for m in Chem.SDMolSupplier(folder_ + f):

            m.SetProp("file" , f)
            writer.write(m)

