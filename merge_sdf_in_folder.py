from rdkit import Chem
import os


folder = "/Users/alexanderhowarth/Desktop/Projects/DM1/TRB0050025/exvol/"

writer = Chem.SDWriter(folder + "combined.sdf")

c=0
c_2 = 0

for f in os.listdir(folder):

    if f.endswith(".sdf") and f != "combined.sdf":

        for m in Chem.SDMolSupplier(folder + f):

            if m:

                writer.write(m)
                c+=1

            if c == 999999:
                c_2+=1
                c = 0
                writer = Chem.SDWriter(folder + "combined_" + str(c_2) + ".sdf")



print(c)