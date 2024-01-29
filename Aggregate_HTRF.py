from rdkit import Chem


file_ = "/Users/alexanderhowarth/Desktop/Projects/YTHDC1/YTHDC1_HTRF_18_model/108_compounds.sdf"

writer = Chem.SDWriter("/Users/alexanderhowarth/Desktop/Projects/YTHDC1/YTHDC1_HTRF_18_model/108_compounds_agg.sdf")

HTRF_column = "ABSOLUTE_IC50"

mol_dict = {}


for m in Chem.SDMolSupplier(file_):

    ID = m.GetProp("ID")

    if ID in list(mol_dict.keys()):

        mol_dict[ID].append(m)

    else:
        mol_dict[ID] = [m]

#take average HTRF

ms = 0

for k,v in zip(mol_dict.keys() , mol_dict.values()):

    HTRF = []

    for m in v:

        htrf = m.GetProp(HTRF_column)
        htrf = htrf.strip(">")
        htrf = htrf.strip("<")
        htrf = float(htrf)

        HTRF.append(htrf)

    av_HTRF = sum(HTRF) / len(HTRF)

    v[0].SetProp("av_HTRF" ,str(av_HTRF))

    writer.write(v[0])
    ms +=1

print("n written " , ms)
