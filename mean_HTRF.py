from rdkit import Chem
import numpy as np


f = "/Users/alexanderhowarth/Desktop/Projects/YTHDC1/DS_231220/Model/HTRF_3_1_2024.sdf"

writer = Chem.SDWriter("/Users/alexanderhowarth/Desktop/Projects/YTHDC1/DS_231220/Model/HTRF_3_1_2024_mean.sdf")

c= 0

prop_dict = {}

for m in Chem.SDMolSupplier(f):

    if m:

        ID_ = m.GetProp("ID")

        if ID_ not in prop_dict.keys():
            try:
                rel = m.GetProp("RELATIVE_IC50")
                rel = rel.replace("<" , "")
                rel = rel.replace(">" , "")
                abs = m.GetProp("ABSOLUTE_IC50")

                abs= abs.replace("<" , "")
                abs = abs.replace(">" , "")

                prop_dict[ID_] = [ m,[float(abs)],[float(rel)] ]

            except:

                print("broken" , ID_)

        else:
            try:


                rel = m.GetProp("RELATIVE_IC50")
                rel = rel.replace("<", "")
                rel = rel.replace(">", "")
                abs = m.GetProp("ABSOLUTE_IC50")

                abs = abs.replace("<", "")
                abs = abs.replace(">", "")

                prop_dict[ID_][1].append(float(abs))
                prop_dict[ID_][2].append(float(rel))

            except:

                print("broken", ID_)


for k,v in zip( prop_dict.keys() , prop_dict.values() ):

    m_ = prop_dict[k][0]

    m_.SetProp("ABSOLUTE_IC50_mean",str( np.mean(prop_dict[k][1]) ))
    m_.SetProp("RELATIVE_IC50_mean",str( np.mean(prop_dict[k][2]) ))

    writer.write(m_)
