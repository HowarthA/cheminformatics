

from rdkit.Chem import PandasTools
from rdkit import Chem
import pandas as pd


df_1 = "/Users/alexanderhowarth/Desktop/Projects/AR/Enumerated/combined_stereo_props_confs_score_agg.sdf"

df_2 = "/Users/alexanderhowarth/Desktop/Projects/AR/Enumerated/combined_stereo_props.sdf"

writer = Chem.SDWriter("/Users/alexanderhowarth/Desktop/Projects/AR/Enumerated/combined_stereo_props_confs_score_agg_out.sdf")

inchi_keys = {}

for m in Chem.SDMolSupplier(df_2):

    inchi_keys[Chem.MolToInchiKey(m)] = { n: m.GetProp(n) for  n in m.GetPropNames() }

c= 0

print(inchi_keys)
b = 0
for m in Chem.SDMolSupplier(df_1):
    try:
        i = Chem.MolToInchiKey(m)

        ps = inchi_keys[i]

        for k,v in zip(ps.keys(), ps.values()) :

            m.SetProp( k,v )

        writer.write(m)
    except:

        b+=1
print(b)