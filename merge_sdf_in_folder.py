from rdkit import Chem
import os


folder = "/Users/alexanderhowarth/Desktop/FP4.5_train/docked_poses/"

writer = Chem.SDWriter(folder + "combined.sdf")

c=0
c_2 = 0

for f in os.listdir(folder):

    if f.endswith(".sdf") and f != "combined.sdf":

        for m in Chem.SDMolSupplier(folder + f):

            if m:

                writer.write(m)
                c+=1





print(c)