from rdkit  import Chem
import numpy as np
import tqdm

import os
import multiprocessing

database = os.path.expanduser("/home/ubuntu/MNT/data/user_data/ahowarth/Enamine_REAL_lead-like_cxsmiles.cxsmiles")

def SearchWorker(args):

    inds = args[0]
    proc = args[1]
    qm = args[2]
    sub_smiles = args[3]

    out_file = open("output_" + str(proc) + ".smi","w")

    with open(database) as file:

        i = 0

        for line in file:

            i+=1

            if i > inds[0] - 1:

                break

        i2= 0

        for line in file:

            smi = line.split()[0]

            try:

                prob_mol = Chem.MolFromSmiles(smi)

            except:

                prob_mol = None

            if prob_mol:

                for s, m in enumerate(qm):

                    if prob_mol.HasSubstructMatch(m):

                        out_file.write(  sub_smiles[s].replace("\n", "") + "\t" + line.replace("\n", "") + "\n"  )

                        break

            i2+=1

            if i2 > len(inds):
                break

        out_file.close()


if __name__ == '__main__':

    #define search fingerprint

    mol_file = "Head_groups_YTHDC1.smiles"

    sub_mols= []

    smiles =[]

    qp = Chem.AdjustQueryParameters()
    qp.makeDummiesQueries = True
    qp.adjustDegree = True
    qp.adjustDegreeFlags = Chem.ADJUST_IGNOREDUMMIES

    for s in open(mol_file,"r").readlines():

        smiles.append(s)
        core_ = Chem.MolFromSmiles(s)
        qm = Chem.AdjustQueryProperties(core_, qp)
        sub_mols.append(qm)

    if os.path.exists("log.txt"):

        completed = [int(l) for l in open("log.txt", "r").readlines()]

    else:

        log_ = open("log.txt", "w+")

        completed = []

    db_length = 0

    with open(database) as file:
        for line in file:
            db_length+=1


    inds = np.arange(0, db_length)

    chunks = 6000

    maxproc = 60

    chunks = np.array_split(inds, chunks)

    args_ = []

    c = 0

    #########

    for i, j in enumerate(chunks):

        if i not in completed:

            args_.append([( j, c,sub_mols,smiles)])

        c += 1

    import multiprocessing

    # defaults to os.cpu_count() workers

    import multiprocessing as mp

   # p = multiprocessing.Pool(maxproc)

    #pbar = tqdm.tqdm(total=len(args))

    #p.map_async(SearchWorker, args,callback=lambda _: pbar.update(1))

    with mp.Pool(maxproc) as p, tqdm.tqdm(total=len(args_)) as pbar:
        res = [p.apply_async(
            SearchWorker,args=args_[i], callback=lambda _: pbar.update(1)) for i in range(len(args_))]
        res_vector = [r.get() for r in res]

    p.close()
    p.join()




