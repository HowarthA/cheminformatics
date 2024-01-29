from Bio import PDB

def read_pdb_file(file_path):
    # Create a PDB parser
    parser = PDB.PDBParser(QUIET=True)

    # Parse the PDB file
    structure = parser.get_structure('protein', file_path)

    return structure

def separate_ligand_and_protein(structure):

    ligand_chain_id = None
    protein_atoms = []
    ligand_atoms = []

    for model in structure:
        print("model" , model)

        for chain in model:
            # Check COMPND field to identify ligand chain

            print("chain " , chain)
            if 'COMPND' in chain.full_id[2][0] and ligand_chain_id is None:
                ligand_chain_id = chain.id
            elif ligand_chain_id is not None and chain.id == ligand_chain_id:
                ligand_atoms.extend(chain.get_atoms())
            else:
                protein_atoms.extend(chain.get_atoms())

    print("protein " , protein_atoms)

    return protein_atoms, ligand_atoms

def save_pdb(structure, atoms, output_path):
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(output_path, select=atoms)

# Replace 'your_file.pdb' with the path to your actual PDB file
pdb_file_path = '/Users/alexanderhowarth/Desktop/test.pdb'


pdb_structure = read_pdb_file(pdb_file_path)

print([c for c in pdb_structure.get_chains()])

print("PDB file successfully loaded.")

protein_atoms, ligand_atoms = separate_ligand_and_protein(pdb_structure)


# Save protein as a separate PDB file
save_pdb(pdb_structure, protein_atoms, '/Users/alexanderhowarth/Desktop/protein.pdb')
print("Protein saved as 'protein.pdb'.")

# Save ligand as an SDF file
with open('/Users/alexanderhowarth/Desktop/ligand.sdf', 'w') as sdf_file:
    PDB.SDWriter(sdf_file).write(ligand_atoms)
print("Ligand saved as 'ligand.sdf'.")

