import os.path

from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolAlign
import tqdm
import numpy as np
import multiprocessing as mp
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

folder = os.path.expanduser( "~/YTHDC1_REAL")

f = folder + "/Enamine_REAL_350_3_lead_like_705_sub_combined"

def embed_mol(mol):

    if type(mol) == str:

        mol = Chem.MolFromSmiles(mol)

    mol = Chem.AddHs(mol, addCoords=True)

    param = rdDistGeom.ETKDGv3()

    param.pruneRmsThresh = 0.2

    n_conformers = 10

    cids = rdDistGeom.EmbedMultipleConfs(mol, n_conformers, param)

    AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0, mmffVariant='MMFF94s')

    Chem.rdMolAlign.AlignMolConformers(mol)

    return mol, cids


def ConfGenWorker(args):

    proc = args[0]

    inds = args[1]

    broken = 0

    i = 0



    sup = Chem.SDMolSupplier(f+ ".sdf")

    mols = []

    final_mols = []

    final_cids = []

    for m in sup:

        if i >= inds[0] and i < inds[-1]:

            mols.append(m)

        i+=1

    for m in mols :

        try:

            mol, cids = embed_mol(m)

            final_mols.append(mol)
            final_cids.append(cids)

        except:

            print("broken" )
            broken+=1

        with Chem.SDWriter(f + "_conformers" + "_proc_" + str(proc) + ".sdf") as w:

            for m,cids in zip(final_mols,final_cids):

                for c in cids:

                    w.write(m, confId=c)

    return broken

if __name__ == '__main__':

    n_mols = len([m for m in Chem.SDMolSupplier(f + ".sdf")])

    print("n_mols = " , n_mols)

    broken = 0

    inds = np.arange(0, n_mols)

    c_n = 600

    maxproc = 60

    chunks = np.array_split(inds, c_n)

    args_ = []

    c = 0

    for i,c in enumerate(chunks):

        args_.append([( i,c) ])

    with mp.Pool(maxproc) as p, tqdm.tqdm(total=len(args_)) as pbar:
        res = [p.apply_async(
            ConfGenWorker, args=args_[i], callback=lambda _: pbar.update(1)) for i in range(len(args_))]
        res_vector = [r.get() for r in res]

    w = Chem.SDWriter( "conformers_total_0" + ".sdf" )

    mol_count = 0

    f_count = 1

    for f2 in os.listdir(folder):

        if "proc" in f2:

            for m in Chem.SDMolSupplier(folder + "/" + f2 ):

                if m:

                    w.write(m)

                    mol_count+=1

                    if mol_count == 999999:

                        mol_count= 0

                        w = Chem.SDWriter("conformers_total_"+ str(f_count) + ".sdf")

                        f_count+=1


        os.remove(folder + "/" + f2)