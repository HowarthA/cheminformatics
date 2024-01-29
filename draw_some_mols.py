from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
from rdkit.Chem import rdFMCS

#the rdkit bible https://www.rdkit.org/docs/GettingStartedInPython.html

# Load the .sdf file

input_file_path = "Path_to_.sdf"
output_file_path = "Path_to_svg.svg"

scale = 300 #in pixels

def draw_one_mol(Input_file_path, Output_file_path):

    #load molecule from file

    mol = Chem.MolFromMolFile(Input_file_path)

    if mol is not None:
        # Generate a 2D depiction of the molecule
        AllChem.Compute2DCoords(mol)


        #drawing magic
        d = rdMolDraw2D.MolDraw2DSVG(scale, scale)
        d.DrawMolecule(mol)
        d.FinishDrawing()
        p = d.GetDrawingText()

        with open(Output_file_path, 'w') as f:
            f.write(p)

        print("SVG image saved.")
    else:
        print("No molecule found in the .sdf file.")

def draw_grid_of_mols(Input_file_path, Output_file_path,align):

    #read in the molecules from an .sdf file

    mols = [ m for m in Chem.SDMolSupplier(Input_file_path) if m  ]

    #options for number of mols per row in the image
    n_per_row = min(5 , len(mols))
    n_rows = int(len(mols) / n_per_row) + (len(mols) % n_per_row > 0)

    #this part of the code tries to find the biggest bit all the molecule sin the file have
    #then it generates 2d coordinates for this common structure
    #then 2d coordinates for all the moleucles are generated sharing the coordinates for the substrucutre

    if align ==True:

        #find the substrcutre
        mcs = rdFMCS.FindMCS(mols, completeRingsOnly=True, ringMatchesRingOnly=True)

        print("largest substrcutre SMARTS is " , mcs)

        #make it into an rdkit mol
        mcs = Chem.MolFromSmarts(mcs.smartsString)

        #generate 2d coordinates
        AllChem.Compute2DCoords(mcs)

        for m in mols:

            try:
                # generate 2d coordinates for all the mols to match
                _ = AllChem.GenerateDepictionMatching2DStructure(m, mcs, allowRGroups=True)

            except:

                AllChem.Compute2DCoords(m)

    #drawing magic

    d2d = rdMolDraw2D.MolDraw2DSVG(n_per_row * scale, n_rows * scale, scale, scale)

    d2d.DrawMolecules(list(mols))

    pic = open(Output_file_path , "w+")

    d2d.FinishDrawing()

    pic.write(str(d2d.GetDrawingText()))

    pic.close()

#example for drawing one mol

draw_one_mol(input_file_path,output_file_path)

#example for drawing multiple mols in a grid

#draw_grid_of_mols(input_file_path,output_file_path,False)

#if you want to draw lots of similar molecules in a grid so that they are all alligned

#draw_grid_of_mols(input_file_path,output_file_path,True)
