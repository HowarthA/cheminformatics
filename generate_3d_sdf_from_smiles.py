from rdkit  import Chem
import numpy as np
import tqdm

import os
import multiprocessing
from rdkit.Chem import rdmolops
import os.path

from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolAlign
import tqdm
import numpy as np
import multiprocessing as mp
from rdkit import RDLogger
import copy
from rdkit.Chem import Descriptors3D
import multiprocessing as mp

input_file = "TL1_2d.sdf"



def embed_(batch,i,mols,IDS):

    output = Chem.SDWriter(input_file[:-4] + "_" + str(i) + "_3d.sdf")

    for ind1 in batch:

        m_ = mols[ind1]

        id_ = IDS[ind1]

        m = copy.deepcopy(m_)

        m = Chem.AddHs(m)

        param = rdDistGeom.ETKDGv3()

        rot_bonds = Chem.rdMolDescriptors.CalcNumRotatableBonds(m)

        if rot_bonds < 8:

            n_conformers = 50

        elif (rot_bonds >= 8) and (rot_bonds <= 12):

            n_conformers = 200

        else:

            n_conformers = 300

        param.pruneRmsThresh = 0.5
        param.useRandomCoords = True
        param.enforceChirality = True
        param.maxAttempts = 10

        cids = rdDistGeom.EmbedMultipleConfs(m, n_conformers, param)

        cids = [c for c in cids]

        AllChem.MMFFOptimizeMoleculeConfs(m, numThreads=0, mmffVariant='MMFF94s')

        mp = AllChem.MMFFGetMoleculeProperties(m, mmffVariant='MMFF94s')

        energies1 = []

        if (mp) and (len(cids) > 1):

            for cid in cids:
                ff = AllChem.MMFFGetMoleculeForceField(m, mp, confId=cid)
                e = ff.CalcEnergy()
                energies1.append(e)

        else:

            energies1.append(0)

        min_e = np.argmin(energies1)

        min_e = cids[min_e]

        Rs = [Descriptors3D.RadiusOfGyration(m, confId=cid) for cid in cids]

        max_R = np.argmax(Rs)

        max_R = cids[max_R]



        m.SetProp("ID" ,id_ )

        output.write(m, confId=max_R)


if __name__ == '__main__':

    mols = [m for m in Chem.SDMolSupplier(input_file) if m ]

    IDS = [ m.GetProp("ID") for m in mols ]

    n_cpus = 64
    n_chunks = 1000

    batches = np.arange(0,len(mols))

    batches = np.array_split(batches, n_chunks)


    with mp.Pool(n_cpus) as p, tqdm.tqdm(total=len(batches)) as pbar:
        res = [p.apply_async(
            embed_, args=[batches[i],i,mols,IDS], callback=lambda _: pbar.update(1)) for i in range(len(batches))]
        res_vector = [r.get() for r in res]


output = Chem.SDWriter( "output.sdf" )

for f in os.listdir():

    if f.endswith("3d.sdf"):

        for m in Chem.SDMolSupplier(f,removeHs= False):

            output.write(m)