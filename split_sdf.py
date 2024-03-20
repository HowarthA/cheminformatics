from rdkit import Chem

file_ = "/Users/alexanderhowarth/Desktop/Projects/YTHDC1/DS/DS_all/DS_comb_confs_53551_ans_agg_out_sol.sdf"

file_name_prop = "DS"

outfolder  = "/Users/alexanderhowarth/Desktop/Projects/YTHDC1/DS/Rescore/"

writers = {}

for m in Chem.SDMolSupplier(file_):

    #m.RemoveAllConformers()

    f = m.GetProp(file_name_prop)[:-4] + "_53551_3d.sdf"

    if f in writers.keys():

        writers[f].write(m)

    else:

        writers[f] = Chem.SDWriter(outfolder + f)

        writers[f].write(m)
