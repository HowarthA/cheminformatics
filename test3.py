from rdkit import Chem



stereo_mols =  [m for m in Chem.SDMolSupplier("/Users/alexanderhowarth/Desktop/Projects/YTHDC1/HTRF_model_0124/HTRF_data_stereo.sdf")]


conf_mols =  [m for m in Chem.SDMolSupplier("/Users/alexanderhowarth/Desktop/Projects/YTHDC1/HTRF_model_0124/conformers.sdf")]

merged_conf_mols =  [m for m in Chem.SDMolSupplier("/Users/alexanderhowarth/Desktop/Projects/YTHDC1/HTRF_model_0124/HTRF_data_confs.sdf")]


IDs = []
V_IDs =[ ]

for m in merged_conf_mols:
    try:
        IDs.append(    m.GetProp("VENDOR_ID"))
    except:

        print("no ID")



print(len(set(IDs)))