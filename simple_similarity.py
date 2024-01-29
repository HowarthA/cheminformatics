from rdkit import Chem
import numpy as np
import tqdm
from rdkit.Chem import AllChem
import os
import multiprocessing
from rdkit import DataStructs

database = "/Users/alexanderhowarth/Downloads/Enamine_REAL_350-3_lead-like_cxsmiles.cxsmiles"

def SearchWorker(args):

    batch_len = 100000

    threshold = 0.7

    inds = args[0]
    proc = args[1]
    search_fps = args[2]

    out_file = open("output_" + str(proc) + ".smi", "w")

    with open(database) as file:

        i = 0

        for line in file:

            i += 1

            if i > inds[0] - 1:
                break

        i2 = 0

        FP_Batch = [ '' for i in range(batch_len) ]

        smiles_chache = ['' for i in range(batch_len)]

        cache_count = 0

        for line in file:

            smi = line.split()[0]

            try:

                prob_mol = Chem.MolFromSmiles(smi)

            except:

                prob_mol = None

            if prob_mol:

                smiles_chache[cache_count] = line

                FP_Batch[cache_count] = AllChem.GetMorganFingerprintAsBitVect(prob_mol, 2, 1024)

            if cache_count == batch_len - 1:

                #calculate similarities

                for fp_n , fp in enumerate(search_fps):

                    sims =  np.array(DataStructs.BulkTanimotoSimilarity(fp, FP_Batch))

                    for sim_n , s in enumerate(sims):

                        if s > threshold:

                            out_file.write( str(fp_n ) + " " + smiles_chache[sim_n] )

                cache_count = 0

            else:

                cache_count+=1

            i2 += 1

            if i2 > len(inds):
                break

        out_file.close()


if __name__ == '__main__':

    # define search fingerprint

    search_sdf = "/Users/alexanderhowarth/Desktop/Projects/EML4-ALK/EML4-ALK_followup/TRB0023760/TRB0023760_all.sdf"

    output = "/Users/alexanderhowarth/Desktop/Projects/EML4-ALK/EML4-ALK_followup/TRB0023760/TRB0023760_all_sim.sdf"

    #ref_mol = "CCOC1CC(N(C)CC(=O)Nc2c(C)n[nH]c2C)C11CCCCC1"

    ref_mol  = "Fc1ccc(cc1)-c1cnc(CN2C(=O)NC3(CCSC3)C2=O)o1"

    ref_mol =Chem.MolFromSmiles(ref_mol)

    ref_fp =  AllChem.GetMorganFingerprintAsBitVect(ref_mol, 3)

    writer = Chem.SDWriter(output)

    for m in Chem.SDMolSupplier(search_sdf):

        fp = AllChem.GetMorganFingerprintAsBitVect(m,3)

        sim = AllChem.DataStructs.TanimotoSimilarity( ref_fp,fp )

        if sim > 0.6:

            m.SetProp("Similartiy" , str(sim))

            writer.write(m)
