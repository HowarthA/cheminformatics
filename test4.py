import numpy as np
import pandas as pd
from rdkit import Chem
import numpy as np


mols = [m for m in Chem.SDMolSupplier("/Users/alexanderhowarth/Desktop/Libraries/TL1_3d.sdf",removeHs = False)]
writer = Chem.SDWriter("/Users/alexanderhowarth/Desktop/Libraries/TL1_3d_sample.sdf")
select = np.random.randint( 0,len(mols) -1 , 1000 )
for i in select:
    writer.write(mols[i])
