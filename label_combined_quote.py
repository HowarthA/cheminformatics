import os

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
import copy

def standardize(mol):
    clean_mol = rdMolStandardize.Cleanup(mol)
    parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
    uncharger = rdMolStandardize.Uncharger()  # annoying, but necessary as no convenience method exists
    uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
    te = rdMolStandardize.TautomerEnumerator()  # idem
    taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)

    return taut_uncharged_parent_clean_mol


quote_ms = [ standardize(m) for m in Chem.SDMolSupplier("/Users/alexanderhowarth/Desktop/YAP_followup/YAP_quote_products.sdf") ]

quote_IDs = []

for m in quote_ms:

    try:

        id = m.GetProp("Query mcule ID")

    except:

        id = None


    quote_IDs.append(id)

#quote_IDs = [ m.GetProp("Quoted Mcule ID")  for m in quote_ms]

print(quote_IDs)

quote_InChikeys = [ Chem.MolToInchiKey(m) for m in quote_ms ]


print(len(quote_ms))

assigned_comps = ['' for i in quote_InChikeys]

#next assign IDs to compounds

for f in ["SDF_1264_MFs.sdf","SDF_3486_MFs.sdf","SDF_4862_MFs.sdf"]:

        label = f[:-4]

        ms = [m for m in Chem.SDMolSupplier("/Users/alexanderhowarth/Desktop/YAP_followup/" + f)]


        IDs = [m.GetProp("mcule ID") for m in ms]

        for i, inc in enumerate(quote_IDs):

            if inc  in IDs:

                assigned_comps[i]= label



for m, series in zip(quote_ms,assigned_comps):

    m.SetProp("series", str(series))

'''with Chem.SDWriter("/Users/alexanderhowarth/Desktop/YAP_followup/labeled_MF_quote.sdf") as w:
    for m in quote_ms:
        w.write(m)

'''
#### count the number from each series.

counts = {}

counts[''] = 0

for s in assigned_comps:

    if len(s) > 0:

        if s not in counts.keys():

            counts[s] =1

        else:
            counts[s] +=1

    else:

        counts[''] +=1

print(counts)