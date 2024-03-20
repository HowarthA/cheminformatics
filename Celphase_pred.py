from rdkit import Chem
from rdkit.Chem import PandasTools
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
import copy
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

df = PandasTools.LoadSDF("/Users/alexanderhowarth/Desktop/Projects/YTHDC1/Data_analysis/All_series_1_data_props.sdf" , embedProps=True).dropna(subset=['CelPhase Absolute IC50 Mean [µM]','HTRF Absolute IC50 Mean'])


d = [ 100,50,30,20,10 ]
drop = []


for i,r in df.iterrows():

    if float(r['CelPhase Absolute IC50 Mean [µM]']) in d:

        drop.append(i)


df['CelPhase Absolute IC50 Mean [µM]'][drop] = 100
print(df.columns)


df = df.drop(drop)
d = [ 30 ]
drop = []


for i,r in df.iterrows():

    if float(r['HTRF Absolute IC50 Mean']) in d:

        drop.append(i)


print(len(df))

for c in df.columns:
    try:

        df[c] = pd.to_numeric(df[ c ])

    except:

        None


df['CelPhase pIC50'] = -np.log10( df['CelPhase Absolute IC50 Mean [µM]'] * 1E-6 )

df['HTRF pIC50'] = -np.log10( df['HTRF Absolute IC50 Mean'] * 1E-6 )

deltas = np.array((df['HTRF pIC50'] - df['CelPhase pIC50']) > 1.5)

print(np.sum(deltas))



def train_RF(train_fps,test_fps,train_y,test_y):

    #rf = RandomForestRegressor(n_estimators=estimators, random_state=42)
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    #rf = LinearRegression()

    #rf = linear_model.Ridge(alpha=0.001)

    #rf = linear_model.ElasticNet(alpha=0.1,max_iter=2000)

    #rf = linear_model.Lasso(alpha=0.001,max_iter=2000)

    train_y = np.array(train_y)

    rf.fit(train_fps, train_y)

    test_pred = rf.predict(test_fps)

    train_pred = rf.predict(train_fps)

    return test_pred , train_pred

def evaluate_model( fps, y_data, model,color):


    bootstraps = 100
    n_fold = 10

    test_ps = np.zeros((bootstraps, len(y_data)))
    test_vs = np.zeros((bootstraps, len(y_data)))

    for n in range(bootstraps):

        kf = KFold(n_splits=n_fold, shuffle=True)

        for train, test in kf.split(fps):

            train_descs = fps[train]
            test_descs = fps[test]

            train_y_data = y_data[train]
            test_y_data = y_data[test]

            test_pred, train_pred = train_RF(train_descs, test_descs, train_y_data, test_y_data)

            test_vs[n][test] = test_y_data
            test_ps[n][test] = test_pred

    test_ps = np.array(test_ps)
    test_vs = np.array(test_vs)

    av_ps = np.mean(test_ps, axis=0)
    av_vs = np.mean(test_vs, axis=0)

    fpr, tpr, thresholds = metrics.roc_curve(av_vs, av_ps, pos_label=1)

    print("auc", metrics.auc(fpr, tpr))

    print(av_ps)

    plt.plot(fpr, tpr)

    plt.show()

    return av_ps

def evaluate_base_model( fps, y_data, model,color):

    bootstraps = 10
    n_fold = 10

    test_ps = np.zeros((bootstraps, len(y_data)))
    test_vs = np.zeros((bootstraps, len(y_data)))

    for n in range(bootstraps):

        kf = KFold(n_splits=n_fold, shuffle=True)

        for train, test in kf.split(fps):

            train_descs = fps[train]
            test_descs = fps[test]

            train_y_data = y_data[train]
            test_y_data = y_data[test]

            #first perform regression

            reg_model = LinearRegression()

            w = train_y_data > 4

            reg_model.fit(np.expand_dims(train_descs[w,0],axis=-1) ,train_y_data[w] )
            test_vs[n][test] = test_y_data

            test_ps[n][test] = reg_model.predict(np.expand_dims(test_descs[:,0],axis=-1))

    test_ps = np.array(test_ps)
    test_vs = np.array(test_vs)

    av_ps = np.mean(test_ps, axis=0)
    av_vs = np.mean(test_vs, axis=0)
    stds = np.std(test_ps, axis=0)


    w = av_vs > 4

    av_ps_ = av_ps[~w]
    av_vs_ = av_vs[~w]
    stds_ = stds[~w]


    av_vs = av_vs[w]
    av_ps = av_ps[w]
    stds = stds[w]


    spr = pearsonr(av_vs, av_ps)

    reg = LinearRegression().fit(np.array([[a] for a in av_vs]), av_ps)

    rmse = mean_squared_error(av_vs, av_ps)

    plt.title("fixed length FP4 Model n = " + str(len(y_data)) + "\n" + str(bootstraps) + " bootstraps" + "\n" + str(
        n_fold) + " fold cross validation" + "\nPearsons = " + str(
        round(spr.statistic, 3)) + "\nRMSE = " + str(round(rmse, 2)))

    colormap = plt.cm.viridis

    for s, p, v in zip(stds_, av_ps_, av_vs_):
        plt.plot([v, v], [p - s, p + s], color="grey", alpha=0.3)

    for v, p in zip(av_vs_, av_ps_):
        plt.plot(v, p, "o", alpha=0.3, color="grey")

    for s, p, v in zip(stds, av_ps, av_vs):
        plt.plot([v, v], [p - s, p + s], color=color, alpha=0.8)

    for v, p in zip(av_vs, av_ps):
        plt.plot(v, p, "o", alpha=0.8, color=color)

    plt.plot([min(av_ps), max(av_ps)], [min(av_ps), max(av_ps)], linestyle=":", color='grey')

    plt.plot([min(av_vs), max(av_vs)],
             [reg.coef_ * min(av_vs) + reg.intercept_, reg.coef_ * max(av_vs) + reg.intercept_],
             color=color)

    plt.show()
    plt.close()

    return av_ps


'''for n in range(n_boot):

    train, test = train_test_split(df, test_size=0.4)

    print(len(train) , len(test_1) , len(test_2))

    model = LinearRegression()

    model.fit(np.array([ [i ] for i in  train['HTRF pIC50']]) ,train['CelPhase pIC50']  )

    preds_1 = model.predict(np.array([ [i ] for i in  test_1['HTRF pIC50']]))

    preds_2 = model.predict(np.array([ [i ] for i in  test_2['HTRF pIC50']]))

    deltas = preds_1 - test_1['CelPhase pIC50']

    print("error 1" , mean_squared_error(test_1['CelPhase pIC50'], preds_1 ) , len(preds_1 ))

    #correction_model =

    #test_1_fps = [ [r['molLogP'] ,r['molLogD']  ,r['molLogS'] , r['nof_HBA'],r['nof_HBD'], r['MolLogPERME']] for i, r in test_1.iterrows() ]
    #test_2_fps = [ [r['molLogP'] ,r['molLogD']  ,r['molLogS'] , r['nof_HBA'],r['nof_HBD'], r['MolLogPERME']] for i, r in test_2.iterrows() ]


    test_1_fps = [ [ r['MolLogPERME']] for i, r in test_1.iterrows() ]
    test_2_fps = [ [ r['MolLogPERME']] for i, r in test_2.iterrows() ]


    rf = RandomForestRegressor(n_estimators=20, random_state=42)
    rf.fit(test_1_fps, deltas)

    deltas_2 = rf.predict(test_2_fps)

    test_preds = preds_2 - deltas_2

    print("error pre" , mean_squared_error(test_2['CelPhase pIC50'], preds_2) , len(preds_2 ))
    print("error 2" , mean_squared_error(test_2['CelPhase pIC50'], test_preds ) , len(test_preds ))

    plt.plot(preds_2 , test_2['CelPhase pIC50'] , "o")
    plt.plot( test_preds , test_2['CelPhase pIC50'] , "o")
    plt.show()

    quit()'''

fps = np.array([[r['HTRF pIC50'], r['MolLogPERME'] ,r['MolLogPERMECACO2'], r['HBD'],r['molPSA'],r['molLogP'],r['molLogD'],r['molLogS']]  for i, r in df.iterrows()])

ps = evaluate_model(fps, deltas,"RF","C3")


for i , p in zip( df.iterrows() , ps ) :

    r = i[1]

    if p >0.5:

        plt.plot(r['HTRF pIC50'] , r['CelPhase pIC50'] , "o",color = "crimson")
    else:

        plt.plot(r['HTRF pIC50'] , r['CelPhase pIC50'] , "o",color = "deepskyblue")

plt.plot([ min( df['CelPhase pIC50'] ) , max(df['CelPhase pIC50']) ], [ min( df['CelPhase pIC50'] ) , max(df['CelPhase pIC50']) ] ,color = "black")

plt.plot([ min( df['CelPhase pIC50'] ) + 1.5 , max(df['CelPhase pIC50']) + 1.5 ], [ min( df['CelPhase pIC50'] )  , max(df['CelPhase pIC50'])  ] ,color = "black")


plt.xlabel("HTRF pIC50")
plt.ylabel("CelPhase pIC50")
plt.show()