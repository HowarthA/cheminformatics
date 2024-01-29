from rdkit import Chem
import numpy as np
import tqdm
from rdkit.Chem import AllChem
import os
import multiprocessing
from rdkit import DataStructs

#database = "/Users/alexanderhowarth/Downloads/Enamine_REAL_350-3_lead-like_cxsmiles.cxsmiles"
database = os.path.expanduser("~/MNT/data/user_data/ahowarth/Enamine_REAL_350-3_lead-like_cxsmiles.cxsmiles")
#database = "/Users/alexanderhowarth/Desktop/IDsearch/Utitled.smi"

#database = os.path.expanduser("~/MNT/data/user_data/ahowarth/Enamine_REAL_lead-like_cxsmiles.cxsmiles")


def SearchWorker(args):

    batch_len = 10000
    threshold = 0.95

    inds = args[0]
    proc = args[1]

    search_smiles = args[2]
    search_smiles = np.array(search_smiles)

    blank_mol = Chem.MolFromSmiles("C")

    blank_mol_fp = AllChem.GetMorganFingerprintAsBitVect(blank_mol, 2, 1024)

    file = open(database,"r")

    i = 0

    if inds[0] != 0:
        for line in file:
            i += 1
            if i > inds[0] - 1:
                break

    can_Batch = [None for i in range(0,batch_len)]
    smiles_chache = [None for i in range(0,batch_len)]

    cache_count = 0

    i2 = 0

    for line_ in file:

        line = line_.replace('\n', '')

        smi = line.split()[0]

        try:

            prob_mol = Chem.MolFromSmiles(smi)

            smiles_chache[cache_count]  = line.replace('\n', '')

            can_Batch[cache_count] = Chem.CanonSmiles(smi)

            cache_count+=1

        except:

            print("broken")

            smiles_chache[cache_count]  = "Error"

            can_Batch[cache_count] = ""

            cache_count+=1


        if cache_count == batch_len :

            #calculate similarities

            smiles_chache = np.array(smiles_chache)

            for  fp_n, fp in enumerate(search_smiles):

                for can_ ,cache in zip(can_Batch ,smiles_chache ):

                    if can_ == fp:

                        print("found " + fp + " " + str(fp_n)+ " "+ cache )

                        if os.path.exists( "output_" + str(proc) + ".smi"):

                            out_file = open("output_" + str(proc) + ".smi", "a+")

                            out_file.write(cache  + " " + str(fp_n)+ " "+ fp + "\n" )

                        else:

                            out_file = open("output_" + str(proc) + ".smi", "a")

                            out_file.write(cache  + " " + str(fp_n)+ " "+ fp + "\n" )

            FP_Batch = [None for i in range(0, batch_len)]
            smiles_chache = [None for i in range(0, batch_len)]

            cache_count = 0

        if i2 > inds[-1] -1:

            smiles_chache = np.array(smiles_chache)

            for i in range(0,batch_len):

                if FP_Batch[i] == None:

                    FP_Batch[i] = blank_mol_fp
                    smiles_chache[i] = "Error"

            for fp_n, fp in enumerate(search_smiles):

                for can_, cache in zip(can_Batch, smiles_chache):

                    if can_ == fp:

                        print("found " + fp + " " + str(fp_n) + " " + cache)

                        if os.path.exists("output_" + str(proc) + ".smi"):

                            out_file = open("output_" + str(proc) + ".smi", "a+")

                            out_file.write(cache + " " + str(fp_n) + " " + fp + "\n")

                        else:

                            out_file = open("output_" + str(proc) + ".smi", "a")

                            out_file.write(cache + " " + str(fp_n) + " " + fp + "\n")

            break

        i2 += 1

    if os.path.exists("output_" + str(proc) + ".smi"):

        out_file.close()

if __name__ == '__main__':

    # define search fingerprint

    search_sdf = "Alex_Fuzzy_pharmacophore_final_priority_list_2.sdf"

    #search_sdf = "/Users/alexanderhowarth/Desktop/IDsearch/Alex_Fuzzy_pharmacophore_final_priority_list_2.sdf"

    search_smiles = [ Chem.CanonSmiles(Chem.MolToSmiles(s)) for s in Chem.SDMolSupplier(search_sdf)]

    if os.path.exists("log.txt"):

        log_ = open("log.txt","w")

        completed = [int(l) for l in open("log.txt", "r").readlines()]

    else:

        log_ = open("log.txt", "w+")

        completed = []

    db_length = 0

    with open(database) as file:
        for line in file:
            db_length += 1
    inds = np.arange(0, db_length)

    chunks = 6000

    maxproc = 60

    chunks = np.array_split(inds, chunks)

    args_ = []

    c = 0

    #########

    for i, j in enumerate(chunks):

        if i not in completed:

            args_.append([(j, c , search_smiles )])

        c += 1

    import multiprocessing

    # defaults to os.cpu_count() workers

    import multiprocessing as mp

    # p = multiprocessing.Pool(maxproc)

    # pbar = tqdm.tqdm(total=len(args))

    # p.map_async(SearchWorker, args,callback=lambda _: pbar.update(1))

    with mp.Pool(maxproc) as p, tqdm.tqdm(total=len(args_)) as pbar:
        res = [p.apply_async(
            SearchWorker, args=args_[i], callback=lambda _: pbar.update(1)) for i in range(len(args_))]
        res_vector = [r.get() for r in res]

    p.close()
    p.join()



