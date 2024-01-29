from rdkit import Chem
from rdkit.Chem import AllChem
from Bio import PDB


def load_ligand_sdf(file_path):
    # Load ligand from SDF file using RDKit
    suppl = Chem.SDMolSupplier(file_path)
    ligand = next(suppl)
    return ligand


def load_protein_pdb(file_path):
    # Load protein from PDB file using BioPython
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', file_path)
    return structure


def save_complex_pdb(protein_structure, ligand_molecule, output_file):
    # Create a combined structure with two chains (protein and ligand)
    combined_structure = PDB.Structure.Structure('complex')

    # Add protein to the combined structure
    for model in protein_structure:
        for chain in model:
            combined_structure.add(chain)


    # Add ligand as a new chain to the combined structure
    ligand_chain = PDB.Chain.Chain('B')
    ligand_residue = PDB.Residue.Residue((' ', 1, ' '), 'LIG', 1)
    ligand_chain.add(ligand_residue)

    pos = ligand_molecule.GetConformer().GetPositions()

    for atom,p in zip(ligand_molecule.GetAtoms() , pos):

        # RDKit atom coordinates are accessed differently

        ligand_residue.add(PDB.Atom.Atom(atom.GetSymbol(), (p[0], p[1], p[2]), 0, 0, ' ', atom.GetIdx(), atom.GetSymbol()))

    combined_structure[0].add(ligand_chain)

    # Save the combined structure to a new PDB file
    io = PDB.PDBIO()
    io.set_structure(combined_structure)
    io.save(output_file)


if __name__ == "__main__":
    # Replace 'ligand.sdf' and 'protein.pdb' with your actual file paths
    ligand_file_path = '/Users/alexanderhowarth/Desktop/cross_dock/crossdocked_pocket10/1A1D_CYBSA_1_341_0/1j0c_A_rec_1j0c_plp_lig_tt_docked_1.sdf'
    protein_file_path = '/Users/alexanderhowarth/Desktop/cross_dock/crossdocked_pocket10/1A1D_CYBSA_1_341_0/1j0c_A_rec_1j0c_plp_lig_tt_docked_1_pocket10.pdb'
    output_file_path = '/Users/alexanderhowarth/Desktop/test.pdb'

    # Load ligand and protein
    ligand = load_ligand_sdf(ligand_file_path)
    protein = load_protein_pdb(protein_file_path)

    # Save the complex as a single PDB file
    save_complex_pdb(protein, ligand, output_file_path)
