from rdkit import Chem
import numpy as np


f = "/Users/alexanderhowarth/Desktop/Projects/YTHDC1/FP4_HTRF_Model/Data/0209_Model.sdf"

writer = Chem.SDWriter("/Users/alexanderhowarth/Desktop/Projects/YTHDC1/FP4_HTRF_Model/Data/0209_Model_mean.sdf")

c= 0

prop_dict = {}

for m in Chem.SDMolSupplier(f):

    if m:

        ID_ = m.GetProp("ID")

        if ID_ not in prop_dict.keys():
            try:

                abs = m.GetProp("HTRF Absolute IC50 Mean [µM]")

                abs= abs.replace("<" , "")
                abs = abs.replace(">" , "")

                prop_dict[ID_] = [ m,[float(abs)]]

            except:

                print("broken" , ID_)

        else:
            try:



                abs = m.GetProp("HTRF Absolute IC50 Mean [µM]")

                abs = abs.replace("<", "")
                abs = abs.replace(">", "")

                prop_dict[ID_][1].append(float(abs))


            except:

                print("broken", ID_)


for k,v in zip( prop_dict.keys() , prop_dict.values() ):

    m_ = prop_dict[k][0]

    m_.SetProp("HTRF Absolute IC50 Mean [µM]",str( np.mean(prop_dict[k][1]) ))


    writer.write(m_)
