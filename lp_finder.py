from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from matplotlib import pyplot as plt

from scipy.spatial.transform import Rotation
p_table = Chem.GetPeriodicTable()
def calculate_atom_vectors(molecule):
    atom_vectors = {}
    for atom in molecule.GetAtoms():
        atom_idx = atom.GetIdx()
        atom_coord = molecule.GetConformer().GetAtomPosition(atom_idx)

        neighbor_vectors = {}
        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            neighbor_coord = molecule.GetConformer().GetAtomPosition(neighbor_idx)
            vector = neighbor_coord - atom_coord
            neighbor_vectors[neighbor_idx] = vector

        atom_vectors[atom_idx] = neighbor_vectors

    return atom_vectors


def calculate_lone_pair_coordinates(molecule, atom):



    atom_idx = atom.GetIdx()

    atom_coord = molecule.GetConformer().GetAtomPosition(atom_idx)

    degree = atom.GetTotalDegree()
    valence = atom.GetTotalValence()

    electrons = p_table.GetNOuterElecs(atom.GetAtomicNum())

    lone_pairs =int(( electrons - valence)/2)

    hybridization = atom.GetHybridization()

    lp_s = None

    #oxygen/sulphur sp3/ like

    print(atom.GetSymbol(),hybridization,valence, lone_pairs)

    if (valence ==2) and (lone_pairs ==2) and (degree ==2)  and (hybridization ==Chem.HybridizationType.SP3):

        neighbor_vectors = []

        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            neighbor_coord = molecule.GetConformer().GetAtomPosition(neighbor_idx)
            vector = neighbor_coord - atom_coord
            neighbor_vectors.append(vector)

        neighbor_vectors = np.array(neighbor_vectors)
        neighbor_vectors_norm = neighbor_vectors / np.linalg.norm(neighbor_vectors, axis=1, keepdims=True)

        print("here 1 ")
        plane =np.cross( neighbor_vectors[0],neighbor_vectors[1] )
        plane = plane/ np.linalg.norm(plane)

        axis =  np.mean(neighbor_vectors_norm ,axis = 0)
        axis = axis / np.linalg.norm(axis)

        r_ab = Rotation.from_rotvec(-1 * np.pi/2 * axis)

        A_rotated= r_ab.apply(neighbor_vectors_norm)

        # Normalize the vectors in A_rotated to maintain unit length

        A_rotated_normalized = A_rotated / np.linalg.norm(A_rotated, axis=1)[:, np.newaxis]

        r_ab_2 = Rotation.from_rotvec(-1 * np.pi * plane)
        A_rotated_2= r_ab_2.apply(A_rotated_normalized)

        # Normalize the vectors in A_rotated to maintain unit length
        lp_s = A_rotated_2 / np.linalg.norm(A_rotated_2, axis=1)[:, np.newaxis]

        lp_s +=atom_coord
        lp_s = list(lp_s)

    #nitrogen sp3 like

    elif (valence == 3) and (lone_pairs == 1) and (hybridization == Chem.rdchem.HybridizationType.SP3):

        neighbor_vectors = []

        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            neighbor_coord = molecule.GetConformer().GetAtomPosition(neighbor_idx)
            vector = neighbor_coord - atom_coord
            neighbor_vectors.append(vector)

        neighbor_vectors = np.array(neighbor_vectors)
        neighbor_vectors_norm = neighbor_vectors / np.linalg.norm(neighbor_vectors, axis=1, keepdims=True)

        axis =  np.mean(neighbor_vectors_norm ,axis = 0)
        axis = axis / np.linalg.norm(axis)

        lp_s =  [-axis + atom_coord]

    #carbonyl SP2 like

    elif (valence == 2) and (degree == 1) and (lone_pairs ==2) and (hybridization == Chem.rdchem.HybridizationType.SP2):

        n1_atom = atom.GetNeighbors()[0]
        n1_pos =  np.array(molecule.GetConformer().GetAtomPosition(n1_atom.GetIdx()))

        n_n_coords = []

        for a in n1_atom.GetNeighbors():

            n_n_coords.append( molecule.GetConformer().GetAtomPosition(a.GetIdx()) )

        n_n_coords = list([ list(i ) for i in  n_n_coords])
        plane = np.cross(n_n_coords[0] -n1_pos  , n_n_coords[1] - n1_pos)
        axis = plane / np.linalg.norm(plane)

        n_vector = n1_pos - atom_coord

        r_ab = Rotation.from_rotvec(-1 * 2*np.pi/3 * axis)

        A_rotated= r_ab.apply(n_vector)

        A_rotated_2= r_ab.apply(A_rotated)

        # Normalize the vectors in A_rotated to maintain unit length

        lp_s = np.array([A_rotated,A_rotated_2])

        lp_s = lp_s/ np.linalg.norm(lp_s, axis=1)[:, np.newaxis]

        lp_s += atom_coord

        lp_s = list(lp_s)

        '''fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        p = molecule.GetConformer().GetPositions()
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], marker="o", color="black", s=20, alpha=1)

        ax.scatter(atom_coord[0], atom_coord[ 1], atom_coord[ 2], marker="o", color="crimson", s=100, alpha=1)
        ax.scatter(axis[0], axis[ 1], axis[ 2], marker="o", color="C3", s=100, alpha=1)

        #ax.scatter(pocket['x'][:, 0], pocket['x'][:, 1], pocket['x'][:, 2], marker="o", color="deepskyblue", s=20,
                  #alpha=1)

        for x_  in n_n_coords:

            ax.scatter(x_[0], x_[ 1], x_[2], marker="o", color="deepskyblue", s=50, alpha=1)



        ax.scatter(lps[:,0],lps[:,1] ,lps[:,2] , marker="o", color="C4", s=50, alpha=1)


        ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
        plt.show()'''

    # sp2 valence 2

    elif (valence == 2) and (lone_pairs == 2) and (hybridization == Chem.rdchem.HybridizationType.SP2):

        neighbor_vectors = []

        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            neighbor_coord = molecule.GetConformer().GetAtomPosition(neighbor_idx)
            vector = neighbor_coord - atom_coord
            neighbor_vectors.append(vector)

        neighbor_vectors = np.array(neighbor_vectors)
        neighbor_vectors_norm = neighbor_vectors / np.linalg.norm(neighbor_vectors, axis=1, keepdims=True)

        axis = np.mean(neighbor_vectors_norm, axis=0)
        axis = axis / np.linalg.norm(axis)

        lp_s = [ - axis + atom_coord ]

        plane = np.cross( neighbor_vectors_norm[0] ,neighbor_vectors_norm[1]  )

        plane = plane/ np.linalg.norm(plane)


        lp_s += [ plane + atom_coord , -plane + atom_coord ]



    return lp_s




def create_and_optimize_conformer(smiles):
    molecule = Chem.MolFromSmiles(smiles)

    if molecule is not None:
        # Generate a single conformer with random coordinates
        molecule = Chem.AddHs(molecule)
        AllChem.EmbedMolecule(molecule, randomSeed=42)

        # Optimize the geometry using MMFF
        AllChem.MMFFOptimizeMolecule(molecule, maxIters=100, nonBondedThresh=100.0)
        lp_coords =[]
        # Loop over the atoms and print their properties, including lone pairs

        for atom in molecule.GetAtoms():

            # Calculate coordinates for lone pairs
            lone_pair_coordinates = calculate_lone_pair_coordinates(molecule, atom)
            if lone_pair_coordinates is not None:
                lp_coords.extend( lone_pair_coordinates)

        print(lp_coords)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        p = molecule.GetConformer().GetPositions()
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], marker="o", color="black", s=20, alpha=1)

        #ax.scatter(pocket['x'][:, 0], pocket['x'][:, 1], pocket['x'][:, 2], marker="o", color="deepskyblue", s=20,
                  #alpha=1)

        c = 0

        print(lp_coords)
        for lp in lp_coords:

            ax.scatter(lp[0], lp[ 1], lp[2], marker="o", color="C" + str(c), s=50, alpha=1)
            c+=1
        ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
        plt.show()

        print(lp_coords)

        return molecule
    else:
        return None


# Example usage
smiles_string = "C[C+](=O[-1])=O[-1]"
conformer_molecule = create_and_optimize_conformer(smiles_string)

# Checking if conformer was successfully created and optimized
if conformer_molecule is not None:
    print("Conformer created and optimized successfully.")
else:
    print("Failed to create conformer.")

