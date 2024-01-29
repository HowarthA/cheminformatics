from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import rdDepictor
from rdkit.Chem import rdFMCS
from rdkit.Chem import AllChem
#rdDepictor.SetPreferCoordGen(True)

def standardize(mol):

    clean_mol = rdMolStandardize.Cleanup(mol)
    parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
    uncharger = rdMolStandardize.Uncharger()  # annoying, but necessary as no convenience method exists
    uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
    te = rdMolStandardize.TautomerEnumerator()  # idem
    taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)

    return taut_uncharged_parent_clean_mol

def draw(mols, labels, file):

    from rdkit.Chem.Draw import rdMolDraw2D

    n_per_row = min(4 , len(mols))

    scale = 300

    n_rows = int(len(mols) / n_per_row) + (len(mols) % n_per_row > 0)

    d2d = rdMolDraw2D.MolDraw2DSVG(n_per_row * scale, n_rows * scale, scale, scale)

    d2d.DrawMolecules(list(mols[0:100]), legends=labels)

    pic = open(file + ".svg", "w+")

    d2d.FinishDrawing()

    pic.write(str(d2d.GetDrawingText()))

    pic.close()

hit_smiles = "O=C(O)c1ccc(COc2c(OC)cc(/C=C(C#N)\C(=O)Nc3ccccc3)cc2)cc1"
full_TRB = 'TRB0005411'

code = "5411"

output_folder = "/Users/alexanderhowarth/Desktop/EML4-ALK_followup/5411_Tanimoto"

initial_hit =  standardize(Chem.MolFromSmiles(hit_smiles))

centre_fp = AllChem.GetMorganFingerprintAsBitVect(initial_hit, 2, 1024)

df = PandasTools.LoadSDF("/Users/alexanderhowarth/Desktop/EML4-ALK_followup/5411_Tanimoto_NNs_no_nitrile.sdf")

ms = [ m for m in df.ROMol  ]

mcs = rdFMCS.FindMCS(ms , completeRingsOnly=True, ringMatchesRingOnly=True)

MCS_mol = Chem.MolFromSmarts(mcs.smartsString)

#MCS_mol = Chem.MolFromSmiles("COC(=O)C(O)C(F)(F)F")

AllChem.Compute2DCoords(MCS_mol)

print(MCS_mol)

for m in ms:

    try:

        _ = AllChem.GenerateDepictionMatching2DStructure(m, MCS_mol,allowRGroups = True)

    except:

        AllChem.Compute2DCoords(m)



draw( ms , [] , output_folder + "/" +code +"nonitrile" )


