
import copy
import os

from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from matplotlib import colormaps as cm

from matplotlib import pyplot as plt

from sklearn import metrics
import numpy as np
import tqdm
from rdkit.Chem import Descriptors
from rdkit.Chem import Crippen
from rdkit.Chem import rdMolDescriptors
import pickle
from rdkit.Chem import PandasTools
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from rdkit import DataStructs
import pandas as pd
from rdkit.Chem.MolStandardize import rdMolStandardize
import multiprocessing as mp
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from rdkit.Chem import PandasTools
from sklearn.preprocessing import StandardScaler
from rdkit.Chem import Descriptors3D
from sklearn.preprocessing import MinMaxScaler
from rdkit.Chem import rdDistGeom
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolAlign

estimators = 50
alpha = 0.05

def minmax(ys):

    max_ = max(ys)
    min_ = min(ys)

    return (ys - min_) / (max_ - min_)


def train_EN(train_fps,test_fps,train_y,test_y):

    rf = linear_model.ElasticNet(alpha=alpha,max_iter=2000)

    train_y = np.array(train_y)

    rf.fit(train_fps, train_y)

    test_pred = rf.predict(test_fps)

    train_pred = rf.predict(train_fps)

    return test_pred , train_pred


def train_RF(train_fps,test_fps,train_y,test_y):

    #rf = RandomForestRegressor(n_estimators=estimators, random_state=42)
    rf = RandomForestClassifier(n_estimators=estimators, random_state=42)
    #rf = LinearRegression()

    #rf = linear_model.Ridge(alpha=0.001)

    #rf = linear_model.ElasticNet(alpha=0.1,max_iter=2000)

    #rf = linear_model.Lasso(alpha=0.001,max_iter=2000)

    train_y = np.array(train_y)

    rf.fit(train_fps, train_y)

    test_pred = rf.predict(test_fps)

    train_pred = rf.predict(train_fps)

    return test_pred , train_pred


def train_SVC(train_fps,test_fps,train_y,test_y):

    #rf = RandomForestRegressor(n_estimators=estimators, random_state=42)
    svm_ = svm.SVC()
    #rf = LinearRegression()

    #rf = linear_model.Ridge(alpha=0.001)

    #rf = linear_model.ElasticNet(alpha=0.1,max_iter=2000)

    #rf = linear_model.Lasso(alpha=0.001,max_iter=2000)

    train_y = np.array(train_y)

    svm_.fit(train_fps, train_y)

    test_pred = svm_.predict(test_fps)

    train_pred = svm_.predict(train_fps)

    return test_pred , train_pred


def descriptor(m):

    c = 0

    mw = Descriptors.MolWt(m)

    TPSA =  rdMolDescriptors.CalcTPSA(m)

    logp = Crippen.MolLogP(m)
    fsp3 = rdMolDescriptors.CalcFractionCSP3(m)

    return [mw,TPSA,logp,fsp3] + [Descriptors3D.InertialShapeFactor(m ,confId = c) ,Descriptors3D.Asphericity(m,confId = c),Descriptors3D.Eccentricity(m,confId = c) ,Descriptors3D.RadiusOfGyration(m,confId = c) ,Descriptors3D.NPR1(m,confId = c),Descriptors3D.NPR2(m,confId = c),Descriptors3D.SpherocityIndex(m,confId = c)]


def evaluate_model(mols, fps, y_data, model,color,title):

    if model == "RF":

        bootstraps = 50
    else:
        bootstraps = 100

    n_fold = 10

    test_ps = np.zeros((bootstraps, len(y_data)))
    test_vs = np.zeros((bootstraps, len(y_data)))

    for n in tqdm.tqdm(range(bootstraps)):

        kf = KFold(n_splits=n_fold, shuffle=True)

        for train, test in kf.split(fps):

            train_descs = fps[train]
            test_descs = fps[test]

            scaler = StandardScaler()

            scaler.fit(train_descs)

            train_descs = scaler.transform(train_descs)
            test_descs = scaler.transform(test_descs)

            train_y_data = y_data[train]
            test_y_data = y_data[test]

            # make predictions
            if model == "RF":

                test_pred, train_pred = train_RF(train_descs, test_descs, train_y_data, test_y_data)

            if model == "SVC":
                test_pred, train_pred = train_SVC(train_descs, test_descs, train_y_data, test_y_data)

            test_vs[n][test] = test_y_data
            test_ps[n][test] = test_pred

    test_ps = np.array(test_ps)
    test_vs = np.array(test_vs)

    av_ps = np.mean(test_ps, axis=0)
    av_vs = np.mean(test_vs, axis=0)

    fpr, tpr, thresholds = metrics.roc_curve(av_vs, av_ps, pos_label=1)

    metrics.auc(fpr, tpr)

    print(av_ps)

    plt.plot(fpr,tpr)

    plt.show()

    test_vs[n][test] = test_y_data
    test_ps[n][test] = test_pred


    test_ps = np.mean(test_ps , axis = 0)

    print(test_ps)

    scaler = StandardScaler()

    scaler.fit(fps)

    s_fps = scaler.transform(fps)

    # make predictions
    if model == "RF":

        model_ = RandomForestClassifier(n_estimators=estimators, random_state=42)

        model_.fit(s_fps, y_data)

    if model == "SVC":

        model_ = svm.SVC()
        model_.fit(fps, y_data)

    return    metrics.auc(fpr, tpr) , model_, scaler, av_ps ,test_ps


def descriptor(m):
    c = 0
    mw = Descriptors.MolWt(m)

    TPSA =  rdMolDescriptors.CalcTPSA(m)

    logp = Crippen.MolLogP(m)
    fsp3 = rdMolDescriptors.CalcFractionCSP3(m)

    return [mw,TPSA,logp,fsp3] + [Descriptors3D.InertialShapeFactor(m ,confId = c) ,Descriptors3D.Asphericity(m,confId = c),Descriptors3D.Eccentricity(m,confId = c) ,Descriptors3D.RadiusOfGyration(m,confId = c) ,Descriptors3D.NPR1(m,confId = c),Descriptors3D.NPR2(m,confId = c),Descriptors3D.SpherocityIndex(m,confId = c)]


def descriptor_gs(m_):

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
    param.maxAttempts=10

    cids = rdDistGeom.EmbedMultipleConfs(m, n_conformers, param)

    cids = [ c for c in cids ]

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

    grounds_state = [Descriptors3D.InertialShapeFactor(m, confId=min_e), Descriptors3D.Asphericity(m, confId=min_e),Descriptors3D.Eccentricity(m, confId=min_e), Descriptors3D.RadiusOfGyration(m, confId=min_e)]

    mw = Descriptors.MolWt(m)

    TPSA =  rdMolDescriptors.CalcTPSA(m)

    logp = Crippen.MolLogP(m)
    fsp3 = rdMolDescriptors.CalcFractionCSP3(m)
    HBA = rdMolDescriptors.CalcNumHBA(m)
    HBD = rdMolDescriptors.CalcNumHBD(m)
    Bcut = rdMolDescriptors.BCUT2D(m)

    return [ rot_bonds , len(cids) / n_conformers ] + [mw,TPSA,logp,fsp3,HBA,HBD] + grounds_state + Bcut



def function_(m, models_, scaler_):

    try:
        fp = descriptor_gs(m)

        test_fps = scaler_.transform([fp])

        p = models_.predict(test_fps)[0]

    except:

        p = ""

    return p


if __name__ == '__main__':

    fps = []
    y_data = []
    mols =[]
    ID_ = []
    c = 0

    '''for  m in tqdm.tqdm( Chem.SDMolSupplier("/Users/alexanderhowarth/Desktop/Projects/YTHDC1/Solubility/v1903_solubility_charges.sdf",removeHs = False)):

        try:

            y = m.GetProp("Solubility [µM]")
            y = y.strip(">")
            y = y.strip("<")

        except:

            y = None

        if y:


            try:
                props = descriptor_gs(m)


                #if np.log10(float(y)) > 0.25:
                if np.log10(float(y)) > 0.3:

                    c+=1
                    y_data.append(1)

                else:

                    y_data.append(0)

                fps.append(props)
                mols.append(m)
                ID_.append(m.GetProp("ID"))

            except:

                print("broken")


    print(c , len(y_data) - c)

    y_data = np.array(y_data)
    fps = np.array(fps)

    #pickle.dump( fps , open("sol_fps.p","wb"))
    #fps  = pickle.load(  open("sol_fps.p","rb"))

    auc,model_,scaler,av_ps,test_ps = evaluate_model(mols,fps, y_data, "RF","C6","joint_EN")

    print(auc)
    dict_ = {}

    for i, v in zip(ID_, test_ps ):
        dict_[i] = v

    pickle.dump(dict_,open("/Users/alexanderhowarth/Desktop/Projects/YTHDC1/CelPhase_Model/Psol_pred_dict.p", "wb"))

    pickle.dump( [auc,model_,scaler ] , open("solubility_m.p","wb"))

    scaler = StandardScaler()

    fps = scaler.fit_transform(fps)

    pca = PCA(n_components=2)

    pca.fit(fps)

    fps_ = pca.transform(fps)

    print(pca.explained_variance_ratio_)

    for y ,fp , p in zip(y_data,fps_,av_ps):

        if y == 1:

            plt.plot( p, fp[0] ,  "o",alpha = 0.8,color = "deepskyblue")

        else:

            plt.plot( p, fp[0],"o",alpha = 0.8,color = "crimson")

    plt.show()'''


    auc,model_,scaler = pickle.load(  open("solubility_m.p","rb"))

    print(auc)


    quit()

    f = "/Users/alexanderhowarth/Desktop/Projects/YTHDC1/Y_swap/swaps/combined_charge.sdf"

    test_mol_df_1 = PandasTools.LoadSDF(f,embedProps=True)

    f_out = f[:-4] + "_sol.sdf"

    test_fps = []
    test_EN_pred = []

    args = []

    for m in Chem.SDMolSupplier(f):

        args.append([m, model_, scaler])

    print(len(args))

    with mp.Pool(8) as p, tqdm.tqdm(total=len(args)) as pbar:
        res = [p.apply_async(function_, args=args[i], callback=lambda _: pbar.update(1)) for i in range(len(args))]
        res_vector = [r.get() for r in res]

    test_mol_df_1['P_sol'] = res_vector

    PandasTools.WriteSDF(test_mol_df_1,f_out,properties=[ c for c in test_mol_df_1.columns if c != "ROMol"],molColName = "ROMol")


