import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt
from rdkit.Chem import Descriptors3D
from rdkit.Chem import rdDistGeom
import tqdm
from sklearn import linear_model

mols = [ m for m in Chem.SDMolSupplier("/Users/alexanderhowarth/Desktop/Projects/YTHDC1/Solubility/v231018_solubility_calc.sdf") ]

measured_sols = []
calc_sols = []
diff = []

for m in mols:

    s = m.GetProp("Solubility [µM]")

    s_calc = float(m.GetProp("molLogS"))

    if "<" in s:

        s = 1E-6
    else:

        s = float(s)

    measured_sols.append(s)

    s =  np.log10( s  )

    calc_sols.append(s_calc)

    diff.append(s_calc - s)


fps = []

for m in tqdm.tqdm(mols):

    '''fp = AllChem.GetMorganFingerprintAsBitVect(m,3,2048)

    pka_mb = m.GetProp("pKa_mb")

    if "<" in pka_mb:

        pka_mb = 0

    pka_mb = float(pka_mb)

    pka_ma = m.GetProp("pKa_ma")

    if ">" in pka_ma:

        pka_ma = 14

    pka_ma = float(pka_ma)'''


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
    param.maxAttempts=50

    cids = rdDistGeom.EmbedMultipleConfs(m, n_conformers, param)
    energies = []
    AllChem.MMFFOptimizeMoleculeConfs(m, numThreads=0, mmffVariant='MMFF94s')
    mp = AllChem.MMFFGetMoleculeProperties(m, mmffVariant='MMFF94s')

    for cid in cids :
        ff = AllChem.MMFFGetMoleculeForceField(m, mp, confId=cid)
        e = ff.CalcEnergy()
        energies.append(e)

    energies = np.array(energies) - np.min(energies)

    descs = []

    for c in cids:

        desc = [Descriptors3D.InertialShapeFactor(m ,confId = c) ,Descriptors3D.Asphericity(m,confId = c),Descriptors3D.Eccentricity(m,confId = c) ,Descriptors3D.RadiusOfGyration(m,confId = c) ,Descriptors3D.NPR1(m,confId = c),Descriptors3D.NPR2(m,confId = c),Descriptors3D.SpherocityIndex(m,confId = c)]
        descs.append(desc)

    #fp =  [ float(m.GetProp("molLogS")),  float(m.GetProp("molLogP")),  float(m.GetProp("molLogD")) , pka_ma, pka_mb  ]

    fps.append(  np.concatenate( (np.average(descs,axis = 0 , weights=energies) ,np.var(descs,axis = 0))  ))

w = np.where(np.array(measured_sols)>1)[0]
w_ = np.where(np.array(measured_sols)<1)[0]

high_sol_fps = np.array(fps)[w]
high_sol_diffs =np.array(diff)[w]

low_sol_fps = np.array(fps)[w_]
low_sol_diffs = np.array(diff)[w_]
kf = KFold(n_splits=len(high_sol_fps), shuffle=False)

test_preds = np.zeros(len(measured_sols))


print(len(measured_sols), len(fps))
print()


for high , low in zip(kf.split(high_sol_fps)   , kf.split(low_sol_fps))   :


    print(high)
    print(low)

    # make predictions

    high_train_fps = high_sol_fps[high[0]]
    high_test_fps = high_sol_fps[high[1]]

    high_train_diffs = high_sol_diffs[high[0]]
    high_test_diffs = high_sol_diffs[high[1]]

    low_train_fps = low_sol_fps[low[0]]
    low_test_fps = low_sol_fps[low[1]]

    low_train_diffs = low_sol_diffs[low[0]]
    low_test_diffs = low_sol_diffs[low[1]]


    train_fps = np.concatenate(( high_train_fps, low_train_fps))

    train_diffs = np.concatenate(( high_train_diffs, low_train_diffs))

    test_fps = np.concatenate(( high_test_fps, low_test_fps))
    test_diffs = np.concatenate(( high_test_diffs, low_test_diffs))

    test_inds = np.concatenate( (w[high[1]] , w_[low[1]]) )

    print("train" , train_diffs)

    rf = RandomForestRegressor(n_estimators=10, random_state=42,)

    #en = linear_model.ElasticNet(alpha=0.01,max_iter=2000)

    weights = np.ones(len(train_fps))

    weights[0] = 10

    rf.fit(train_fps, train_diffs,sample_weight=weights)

    test_pred = rf.predict(test_fps)

    test_preds[test_inds] = test_pred


print(test_preds)

log_measured = np.log10(np.array(measured_sols))

test_vs =   calc_sols - test_preds

plt.xlabel("measured LogS")
plt.ylabel("calculated LogS")
plt.plot(  log_measured,calc_sols ,"o" ,color = "crimson",label = "molsoft prediction",alpha = 0.8)
plt.plot(log_measured,test_vs,"o",color = "deepskyblue",label = "model prediction",alpha = 0.8)
plt.legend()
plt.show()

print()

quit()


test_ps = np.array(test_ps)
test_vs = np.array(test_vs)

print(test_ps)

av_ps = np.mean(test_ps,axis = 0)
av_vs = np.mean(test_vs,axis = 0)

stds = np.std(test_ps,axis= 0)

spr = spearmanr(av_vs, av_ps)

print(spr)

rmse = mean_squared_error(av_vs, av_ps)

rmse_med = mean_squared_error(y_data, [np.median(y_data) for j in y_data])

r2 = r2_score(av_vs, av_ps)

reg = LinearRegression().fit(np.array([[a] for a in av_vs]), av_ps)

rmse = mean_squared_error(av_vs, av_ps)

std_errors = np.std([abs(v - p) for v, p in zip(av_vs, av_ps)])

plt.title("MF RF Model n = " + str(len(y_data)) + "\n" + str(bootstraps) + " bootstraps" + "\n" + str(n_fold) + " fold cross validation" + "\n" +str(estimators) + " RF estimators")

for s , p , v in zip(stds , av_ps,av_vs):

    plt.plot([v,v],[p - s , p+s], color = "C1",alpha = 0.8)

plt.plot(av_vs, av_ps, "o",
         label="R2 = " + str(round(r2, 2)) + "\nSpearmans rank = " +str(round(spr.statistic,3)) + "\nRMSE = " + str(
             round(rmse, 2)) ,
         alpha=1,color = "C1")
plt.plot([min(av_vs), max(av_vs)], [min(av_vs), max(av_vs)], linestyle=":", color='grey')

plt.plot([min(av_vs), max(av_vs)],
         [reg.coef_ * min(av_vs) + reg.intercept_, reg.coef_ * max(av_vs) + reg.intercept_],
         color="C1")

plt.legend(

)

plt.xlabel("Experimental  -log(IC50 uM)")
plt.ylabel("Predicted -log(IC50 uM)")

# plt.savefig(folder + "/" + r['ID'] + ".png")

plt.show()
plt.close()


