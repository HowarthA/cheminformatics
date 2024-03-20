import sys
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import multiprocessing as mp
from sklearn.ensemble import GradientBoostingRegressor
from rdkit.Chem import rdDepictor
rdDepictor.SetPreferCoordGen(True)
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import rdDepictor
rdDepictor.SetPreferCoordGen(True)
import os
import numpy as np
from rdkit.Chem import rdDepictor
rdDepictor.SetPreferCoordGen(True)
import tqdm
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
import copy
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from morfeus import XTB
from morfeus import Dispersion
from morfeus import SASA
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors3D
from rdkit.Chem import Descriptors
from rdkit.Chem import Crippen
from rdkit.Chem import rdDistGeom
from rdkit.Chem import AllChem
import  pickle
from rdkit.Chem import PandasTools
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn import svm
from sklearn import metrics

def descriptor_gs(m_):

    m = copy.deepcopy(m_)

    param = rdDistGeom.ETKDGv3()

    rot_bonds = Chem.rdMolDescriptors.CalcNumRotatableBonds(m)

    param.pruneRmsThresh = 0.5
    param.useRandomCoords = True
    param.enforceChirality = True
    param.maxAttempts=10

    mp = AllChem.MMFFGetMoleculeProperties(m, mmffVariant='MMFF94s')

    ff = AllChem.MMFFGetMoleculeForceField(m, mp, confId=0)

    e = ff.CalcEnergy()

    docked = [Descriptors3D.InertialShapeFactor(m_, confId=0), Descriptors3D.Asphericity(m_, confId=0),Descriptors3D.Eccentricity(m_, confId=0), Descriptors3D.RadiusOfGyration(m_, confId=0)]

    TPSA =  rdMolDescriptors.CalcTPSA(m)

    logp = Crippen.MolLogP(m)
    fsp3 = rdMolDescriptors.CalcFractionCSP3(m)

    return [e , rot_bonds  ] + [TPSA,logp,fsp3] + docked

def match_to_substructures(mol):

    HDonorSmarts = Chem.MolFromSmarts('[$([N;!H0;v3]),$([N;!H0;+1;v4]),$([O,S;H1;+0]),$([n;H1;+0])]')
    # changes log for HAcceptorSmarts:
    #  v2, 1-Nov-2008, GL : fix amide-N exclusion; remove Fs from definition

    HAcceptorSmarts = Chem.MolFromSmarts('[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),' +
                                         '$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=!@[O,N,P,S])]),' +
                                         '$([nH0,o,s;+0])]')

    # find any atoms that match these macs keys

    donor_atoms = [i[0] for i in mol.GetSubstructMatches(HDonorSmarts, uniquify=1)]

    acceptor_atoms = [i[0] for i in mol.GetSubstructMatches(HAcceptorSmarts, uniquify=1)]

    is_donor_atom = []

    is_acceptor_atom = []

    for atom in mol.GetAtoms():

        id = atom.GetIdx()

        if id in donor_atoms:

            is_donor_atom.append(True)

        else:
            is_donor_atom.append(False)

        if id in acceptor_atoms:

            is_acceptor_atom.append(True)

        else:

            is_acceptor_atom.append(False)

    is_donor_atom = np.array(is_donor_atom)
    is_acceptor_atom = np.array(is_acceptor_atom)

    return is_donor_atom, is_acceptor_atom

def make_mol(mol, id):
    all_coords = []
    all_masses = []
    all_aromatic = []
    all_atom_number = []

    positions = mol.GetConformer(id).GetPositions()

    for atom, p in zip(mol.GetAtoms(), positions):

        all_masses.append(atom.GetMass())
        all_atom_number.append(atom.GetAtomicNum())
        all_aromatic.append(atom.GetIsAromatic())
        all_coords.append(p)

    return np.array(all_masses), np.array(all_coords), np.array(all_atom_number), np.array(all_aromatic)

def sigmoid_function(x,R,k):
    return (1 / (1 + np.exp(k * (x - R))))

def make_representation(m, i):

    atom_masses, atom_coords, atomic_nums, atom_aromatic = make_mol(m, i)

    is_donor_atom, is_acceptor_atom = match_to_substructures(m)

    ############# next job is to "color the beads in"

    # define some atomwise properties

    ComputeGasteigerCharges(m)

    charges = []

    for a in m.GetAtoms():
        charges.append(float(a.GetProp("_GasteigerCharge")))

    formal_charges = []

    for a in m.GetAtoms():
        formal_charges.append(a.GetFormalCharge())

    formal_charges = np.array(formal_charges)

    xtb = XTB(atomic_nums, atom_coords)

    c_ = xtb.get_charges()

    xtb_charges = np.array([c_[k] for k in sorted(c_.keys())])

    charges = np.array(charges)

    dispersion = Dispersion(atomic_nums, atom_coords)

    atom_p_int_ = dispersion.atom_p_int

    atom_p_int = np.array([atom_p_int_[k] for k in sorted(atom_p_int_.keys())])

    atom_areas_ = dispersion.atom_areas
    atom_areas = np.array([atom_areas_[k] for k in sorted(atom_areas_.keys())])

    sasa = SASA(atomic_nums, atom_coords)
    atom_sa = sasa.atom_areas
    atom_sa = np.array([atom_sa[k] for k in sorted(atom_sa.keys())])

    nucleophilicity_ = xtb.get_fukui('nucleophilicity')

    nucleophilicity = np.array([nucleophilicity_[k] for k in sorted(nucleophilicity_.keys())])

    electrophilicity_ = xtb.get_fukui('electrophilicity')

    electrophilicity = np.array([electrophilicity_[k] for k in sorted(electrophilicity_.keys())])

    sasa = SASA(atomic_nums, atom_coords)

    atom_vo = sasa.atom_volumes

    atom_vo = np.array([atom_vo[k] for k in sorted(atom_vo.keys())])

    charges = np.array(charges)

    CC = np.array(rdMolDescriptors._CalcCrippenContribs(m))

    logP_c = CC[:, 0]

    ###### next make the representation

    # find distances to atoms from beads

    ####

    rep_ = np.vstack((

        charges,
        formal_charges,
        xtb_charges,
        atom_masses,
        logP_c,
        atom_vo,
        atom_aromatic,
        is_donor_atom,
        is_acceptor_atom,
        atom_p_int,
        atom_sa,
        atom_areas,
        nucleophilicity,
        electrophilicity,
    )
    )

    return rep_

def make_representation_2(m, i):

    atom_masses, atom_coords, atomic_nums, atom_aromatic = make_mol(m, i)

    is_donor_atom, is_acceptor_atom = match_to_substructures(m)

    ############# next job is to "color the beads in"

    # define some atomwise properties

    ComputeGasteigerCharges(m)

    charges = []

    for a in m.GetAtoms():
        charges.append(float(a.GetProp("_GasteigerCharge")))

    formal_charges = []

    for a in m.GetAtoms():
        formal_charges.append(a.GetFormalCharge())

    formal_charges = np.array(formal_charges)

    xtb = XTB(atomic_nums, atom_coords)

    c_ = xtb.get_charges()

    xtb_charges = np.array([c_[k] for k in sorted(c_.keys())])

    charges = np.array(charges)

    dispersion = Dispersion(atomic_nums, atom_coords)

    atom_p_int_ = dispersion.atom_p_int

    atom_p_int = np.array([atom_p_int_[k] for k in sorted(atom_p_int_.keys())])

    atom_areas_ = dispersion.atom_areas
    atom_areas = np.array([atom_areas_[k] for k in sorted(atom_areas_.keys())])

    sasa = SASA(atomic_nums, atom_coords)
    atom_sa = sasa.atom_areas
    atom_sa = np.array([atom_sa[k] for k in sorted(atom_sa.keys())])

    nucleophilicity_ = xtb.get_fukui('nucleophilicity')

    nucleophilicity = np.array([nucleophilicity_[k] for k in sorted(nucleophilicity_.keys())])

    electrophilicity_ = xtb.get_fukui('electrophilicity')

    electrophilicity = np.array([electrophilicity_[k] for k in sorted(electrophilicity_.keys())])

    sasa = SASA(atomic_nums, atom_coords)

    atom_vo = sasa.atom_volumes

    atom_vo = np.array([atom_vo[k] for k in sorted(atom_vo.keys())])

    charges = np.array(charges)

    CC = np.array(rdMolDescriptors._CalcCrippenContribs(m))

    logP_c = CC[:, 0]

    ###### next make the representation

    # find distances to atoms from beads

    ####

    rep_ = np.vstack((

        charges,
        formal_charges,
        xtb_charges,
        logP_c,
        atom_vo,
        atom_aromatic,
        is_donor_atom,
        is_acceptor_atom,
        atom_p_int,
        atom_sa,
        atom_areas,

    )
    )

    return rep_


def flatten_last_two_dimensions(arr):
    shape = arr.shape[:-2] + (-1,)
    return arr.reshape(shape)

def find_points_2(pos, reps, R,k):

    t_b_n = 0

    all_pos = np.array([])
    all_reps = np.array([])
    m_inds = np.array([])

    n_mols = 0

    for pos_, rep_ in zip(pos, reps):

        if len(all_pos) == 0:

            all_pos = copy.copy(pos_)
            all_reps = copy.copy(rep_)
            m_inds = np.zeros(len(pos_))

        else:

            all_pos = np.vstack((all_pos, pos_))
            all_reps = np.hstack((all_reps, rep_))
            m_inds = np.hstack((m_inds, np.ones(len(pos_)) * n_mols))

        n_mols += 1

    dists = pairwise_distances(all_pos, all_pos)

    D_weights = sigmoid_function(dists , R, k)

    all_reps = all_reps.T

    '''sum_matrix = np.zeros((len(all_pos), n_mols))

    i = 0

    for test_pos in all_pos:

        for j in range(n_mols):

            D_w = D_weights[i, m_inds == j]

            sum_matrix[i, j] = np.sum(D_w * all_reps[p, m_inds == j])/np.sum(D_w)

        i += 1'''

    sum_matrix = np.zeros((len(all_pos), n_mols, 14))

    for j in range(n_mols):

        D_w = D_weights[:, m_inds == j]

        #original
        #sum_matrix[:, j,:] = np.sum(D_w[:, :, np.newaxis] * all_reps[m_inds == j,:], axis=1)

        #weighted

        sum_matrix[:, j,:] = np.sum(D_w[:, :, np.newaxis] * all_reps[m_inds == j,:], axis=1) * np.sum(D_w[:, :, np.newaxis], axis=1)

    # Compute standard deviation across rows
    sum_std = np.std(sum_matrix, axis=1)

    # Compute the mean of the standard deviations
    sum_std_mean = np.mean(sum_std , axis = 0)

    total_beads = []

    for p in range(0,14):

        beads =[]
        stds_ = copy.copy(sum_std.T[p])
        pos_ = copy.copy(all_pos)
        dists_ = copy.copy(dists)
        max_std = sum_std_mean[p] + 1

        while (len(pos_) > 0) & (max_std > sum_std_mean[p]):

            max_point = np.argmax(stds_)

            max_std = stds_[max_point]
            max_pos = pos_[max_point]

            if max_std > sum_std_mean[p]:
                beads.append(max_pos)
                w = dists_[max_point, :] > R
                dists_ = dists_[w, :][:, w]
                stds_ = stds_[w]
                pos_ = pos_[w]

                # recalculate std?
                # sum_std_mean = np.mean(stds_)

        total_beads.append(np.array(beads))

        t_b_n += len(beads)

    return total_beads

def find_points_3(pos, reps, R,k,y,alpha,prune):

    t_b_n = 0

    all_pos = np.array([])
    all_reps = np.array([])
    m_inds = np.array([])

    n_mols = 0

    for pos_, rep_ in zip(pos, reps):

        if len(all_pos) == 0:

            all_pos = copy.copy(pos_)
            all_reps = copy.copy(rep_)
            m_inds = np.zeros(len(pos_))

        else:

            all_pos = np.vstack((all_pos, pos_))
            all_reps = np.hstack((all_reps, rep_))
            m_inds = np.hstack((m_inds, np.ones(len(pos_)) * n_mols))

        n_mols += 1

    dists = pairwise_distances(all_pos, all_pos)

    D_weights = sigmoid_function(dists , R, k)

    all_reps = all_reps.T

    '''sum_matrix = np.zeros((len(all_pos), n_mols))

    i = 0

    for test_pos in all_pos:

        for j in range(n_mols):

            D_w = D_weights[i, m_inds == j]

            sum_matrix[i, j] = np.sum(D_w * all_reps[p, m_inds == j])/np.sum(D_w)

        i += 1'''

    sum_matrix = np.zeros(( n_mols,len(all_pos), 14))

    for j in range(n_mols):

        D_w = D_weights[:, m_inds == j]

        #original
        #sum_matrix[:, j,:] = np.sum(D_w[:, :, np.newaxis] * all_reps[m_inds == j,:], axis=1)

        #weighted
        sum_matrix[j,:,:] = np.sum(D_w[:, :, np.newaxis] * all_reps[m_inds == j,:], axis=1) * np.sum(D_w[:, :, np.newaxis], axis=1)
        #varience
        #sum_matrix[:, j,:] = np.var( D_w[:, :, np.newaxis] * all_reps[m_inds == j,:], axis=1)

        #sum_matrix[:, j,:] = np.sum(D_w[:, :, np.newaxis] * all_reps[m_inds == j,:], axis=1) / np.sum(D_w, axis=1)[:,None]

    sum_matrix = flatten_last_two_dimensions(sum_matrix)

    scaler = StandardScaler()

    sum_matrix = scaler.fit_transform(sum_matrix)

    mean_y = np.mean(y)
    std_y = np.std(y)

    y = (y -mean_y) / std_y

    lsvc = linear_model.ElasticNet(alpha = alpha).fit(sum_matrix, y)

    model = SelectFromModel(lsvc, prefit=True)

    support = model.get_support(indices=True)

    total_beads = []

    for i in range(0,14):

        w = (support % 14) == i

        bead_pos = all_pos[  (support // 14 )[ w ]]

        if len(bead_pos) ==1:

            total_beads.append( bead_pos)

        elif len(bead_pos) > 1:

            #do very small distance pruning

            bead_dists = pairwise_distances(bead_pos, bead_pos)

            AggCluster = AgglomerativeClustering(distance_threshold=prune, n_clusters=None, metric='precomputed',
                                                 linkage='average')

            clusters = AggCluster.fit(bead_dists)

            #for each cluster pick the biggest coefficient

            cluster_ids = clusters.labels_

            beads =[]

            for c in range(0,len(set(cluster_ids))):

                w1 = np.where(cluster_ids ==c)[0][0]
                beads.append(bead_pos[w1])

            total_beads.append( np.array(beads) )

        else:

            total_beads.append(np.array([]))

    return total_beads

def find_points_4(pos, reps, R,k):

    t_b_n = 0

    all_pos = np.array([])
    all_reps = np.array([])
    m_inds = np.array([])

    n_mols = 0

    for pos_, rep_ in zip(pos, reps):

        if len(all_pos) == 0:

            all_pos = copy.copy(pos_)
            all_reps = copy.copy(rep_)
            m_inds = np.zeros(len(pos_))

        else:

            all_pos = np.vstack((all_pos, pos_))
            all_reps = np.hstack((all_reps, rep_))
            m_inds = np.hstack((m_inds, np.ones(len(pos_)) * n_mols))

        n_mols += 1

    # Compute the mean of the standard deviations

    total_beads = []

    for p in range(0,11):
        m_inds_ = copy.copy(m_inds)
        sum_matrix = np.zeros((len(all_pos), n_mols))
        dists = pairwise_distances(all_pos, all_pos)
        D_weights = sigmoid_function(dists, R, k)

        for j in range(n_mols):
            D_w = D_weights[:, m_inds == j]
            sum_matrix[:, j] = np.sum(D_w[:,: ] * all_reps[ p,m_inds == j].T, axis=1)


        # Compute standard deviation across rows
        sum_std = np.std(sum_matrix, axis=1)

        stds_ = copy.copy(sum_std.T)

        sigma_total = np.std(stds_)
        sigma_mean = np.mean(stds_)



        beads =[]
        pos_ = copy.copy(all_pos)

        D_weights_ = copy.copy(D_weights)

        all_reps_ = copy.copy(all_reps[p,:])

        max_std = (sigma_total  ) + sigma_mean + 1

        while (len(all_reps_) > 0) & (max_std > (( 1.5*sigma_total  )  +sigma_mean) ):

            sigma_total = np.std(stds_)
            sigma_mean  = np.mean(stds_)
            max_point = np.argmax(stds_)

            max_std = stds_[max_point]
            max_pos = pos_[max_point]

            if (max_std > ((1*sigma_total  )  + sigma_mean) ):

                beads.append(max_pos)
                w = D_weights_[max_point,:] < 0.5
                D_weights_ = D_weights_[w,:][:,w]

                pos_ = pos_[w]
                all_reps_ = all_reps_[w]
                m_inds_ = m_inds_[w]
                sum_matrix = np.zeros((len(pos_), n_mols))

                for j in range(n_mols):

                    D_w = D_weights_[:, m_inds_ == j]

                    sum_matrix[:, j] = np.sum(D_w[:, :] * all_reps_[ m_inds_ == j].T, axis=1)

                stds_= np.std(sum_matrix, axis=1).T

                sigma_total = np.std(stds_)
                sigma_mean = np.mean(stds_)

                # recalculate std?
                # sum_std_mean = np.mean(stds_)

        total_beads.append(np.array(beads))

        t_b_n += len(beads)

    return total_beads

def find_points_random(pos, reps, R,k):

    t_b_n = 0

    all_pos = np.array([])
    all_reps = np.array([])
    m_inds = np.array([])

    n_mols = 0

    for pos_, rep_ in zip(pos, reps):

        if len(all_pos) == 0:

            all_pos = copy.copy(pos_)
            all_reps = copy.copy(rep_)
            m_inds = np.zeros(len(pos_))

        else:

            all_pos = np.vstack((all_pos, pos_))
            all_reps = np.hstack((all_reps, rep_))
            m_inds = np.hstack((m_inds, np.ones(len(pos_)) * n_mols))

        n_mols += 1

    # Compute the mean of the standard deviations

    total_beads = []

    for p in range(0,11):

        n_beads = np.random.randint(0,10)

        inds = np.random.randint(0,len( all_pos) -1 , size = n_beads)

        total_beads.append(all_pos[inds])

    return total_beads


def make_representation_morfeus_var_gpt(beads, m, i,R,k,rep_=None):

    atom_coords = m.GetConformer(i).GetPositions()

    if rep_ is None:
        rep_ = make_representation_2(m, i)

    final_rep = np.array([])

    for bead_type, rep in zip(beads, rep_):

        if len(bead_type) > 0:

            bead_dists_to_atoms = pairwise_distances(  atom_coords, bead_type)

            #bead_to_atoms_unit_vectors = (bead_type - atom_coords[:, None]) / bead_dists_to_atoms[..., None]

            weights = sigmoid_function(bead_dists_to_atoms, R, k)

            #weights_vc = weights[:,:, None] * bead_to_atoms_unit_vectors

            w_weights = weights > 0.5

            rep_w = rep[:,None] * w_weights

            max_R =  np.nanmax( rep_w,axis=0)
            min_R = np.nanmin( rep_w,axis=0)

            max_R[ np.isnan(max_R) ] = 0
            min_R[ np.isnan(min_R) ] = 0

            sum_R = np.sum(rep[:,None] * weights , axis=0)

            #vec_R = np.sum(rep[:,None,None] * weights_vc , axis= 0).flatten()

            #final_rep = np.hstack((final_rep,sum_R ,max_R,min_R , vec_R))

            final_rep = np.hstack((final_rep,sum_R ,max_R,min_R ))
            final_rep = np.hstack((final_rep,sum_R ))

    final_rep = np.array(final_rep)

    final_rep[np.isnan(final_rep)] = 0

    return final_rep

def train_EN(train_fps, test_fps, val_fps, train_y,  val_y):

    alpha = [0.01,0.05,0.1]
    l1_ratio = [0.1,0.5,0.8]

    #l1_ratio = [0.5]
    prs = []
    rmse = []
    models = []

    for a  in alpha:

        for l1 in l1_ratio:

            en = linear_model.ElasticNet(alpha=a, l1_ratio=l1,  max_iter=2000)

            train_y = np.array(train_y)

            en.fit(train_fps, train_y)

            val_pred = en.predict(val_fps)

            prs_ = pearsonr(val_y, val_pred)

            rmse_ = mean_squared_error(val_y, val_pred)

            prs.append(prs_.statistic)

            rmse.append(rmse_)

            models.append(en)

    i = np.argmax(prs)

    #i = np.argmin(rmse)

    en = models[i]

    test_pred = en.predict(test_fps)

    train_pred = en.predict(train_fps)

    val_pred = en.predict(val_fps)

    return test_pred, train_pred, val_pred

def train_RF(train_fps, test_fps, val_fps, train_y,  val_y):

    rf = RandomForestRegressor(n_estimators=50, random_state=42)

    train_y = np.array(train_y)

    rf.fit(train_fps, train_y)

    test_pred = rf.predict(test_fps)

    train_pred = rf.predict(train_fps)

    val_pred = rf.predict(val_fps)

    return test_pred, train_pred, val_pred


def train_SVC(train_fps, test_fps, val_fps, train_y, val_y):

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

    val_pred = svm_.predict(val_fps)


    return test_pred , train_pred , val_pred

def train_GBR(train_fps, test_fps, val_fps, train_y, val_y):

    max_depth = [2,1,None]

    n_est = [500,200]

    lr = [0.01,0.1]

    prs = []
    rmse = []
    models = []


    for lr_ in lr:

        for n_est_ in n_est:

            for max_depth_ in max_depth:

                reg = GradientBoostingRegressor(random_state=42, learning_rate=lr_, n_estimators=n_est_,
                                                max_depth=max_depth_)

                train_y = np.array(train_y)

                reg.fit(train_fps, train_y)

                val_pred = reg.predict(val_fps)

                prs_ = pearsonr(val_y, val_pred)

                rmse_ = mean_squared_error(val_y, val_pred)

                prs.append(prs_.statistic)

                rmse.append(rmse_)

                models.append(reg)

    #i = np.argmax(prs)

    i = np.argmin(rmse)

    reg = models[i]

    test_pred = reg.predict(test_fps)

    train_pred = reg.predict(train_fps)

    val_pred = reg.predict(val_fps)

    return test_pred, train_pred, val_pred


def train_XG(train_fps, test_fps, val_fps, train_y,  val_y):

    max_depth = [2,1]

    n_est = [500,200]

    lr = [0.01,0.1]

    prs = []
    rmse = []
    models = []
    train_y = np.array(train_y)

    for lr_ in lr:

        for n_est_ in n_est:

            for max_depth_ in max_depth:

                # specify parameters via map
                param = {'booster': 'dart',
                         'max_depth': max_depth_,
                         'learning_rate': lr_,
                         'objective': 'binary:logistic',
                         'sample_type': 'uniform',
                         'normalize_type': 'tree',
                         'rate_drop': 0.1,
                         'skip_drop': 0.5}

                num_round = n_est_
                bst = xgb.train(param, train_fps, num_round)

                val_pred = bst.predict(val_fps)

                prs_ = pearsonr(val_y, val_pred)

                rmse_ = mean_squared_error(val_y, val_pred)

                prs.append(prs_.statistic)

                rmse.append(rmse_)

                models.append(bst)

    #i = np.argmax(prs)

    i = np.argmin(rmse)

    reg = models[i]

    test_pred = reg.predict(test_fps)

    train_pred = reg.predict(train_fps)

    val_pred = reg.predict(val_fps)

    return test_pred, train_pred, val_pred



def train_GBR_hyper(train_fps, val_fps, train_y , hyper_params):


    lr_ = hyper_params['lr']
    n_est_ = hyper_params['n_est']
    max_depth_ = hyper_params['max_depth']


    reg = GradientBoostingRegressor(random_state=42, learning_rate=lr_, n_estimators=n_est_,
                                    max_depth=max_depth_)

    train_y = np.array(train_y)

    reg.fit(train_fps, train_y)

    train_pred = reg.predict(train_fps)

    val_pred = reg.predict(val_fps)

    return   val_pred ,reg


def train_EN_hyper(train_fps, val_fps, train_y , hyper_params):


    alpha = hyper_params['alpha']
    l1 = hyper_params['l1_ratio']

    reg = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=2000)

    train_y = np.array(train_y)

    reg.fit(train_fps, train_y)

    train_pred = reg.predict(train_fps)

    val_pred = reg.predict(val_fps)

    return   val_pred ,reg


def bootstrap(mols,y_data,full_reps_1,full_reps_2,R1,R2,k1,k2,model,prop_descs):

    kf = KFold(n_splits=10, shuffle=True)

    test_ps = np.zeros( len(y_data))

    for train, test in kf.split(mols):

        inds = np.arange(len(train))

        val_inds = np.random.choice(inds, size=int(len(inds) * 0.2), replace=False)

        train_inds = np.sort(list(set(inds) - set(val_inds)))

        train_mols = mols[train[train_inds]]
        val_mols = mols[train[val_inds]]

        train_y = y_data[train[train_inds]]
        val_y = y_data[train[val_inds]]

        full_train_reps_1 = [full_reps_1[train[i]] for i in train_inds]
        full_train_reps_2 = [full_reps_2[train[i]] for i in train_inds]

        test_mols = [mols[i] for i in test]
        test_y = [y_data[i] for i in test]

        #####

        pos_1 = []
        pos_2 = []

        for m in train_mols:

            pos_1.append(m[0].GetConformer().GetPositions())
            pos_2.append(m[1].GetConformer().GetPositions())

        total_beads_1 = find_points_4(pos_1, full_train_reps_1, R1, k1)
        total_beads_2 = find_points_4(pos_2, full_train_reps_2, R1, k1)

        #total_beads_1 = find_points_2(pos_1, full_train_reps_1, R1, k1)
        #total_beads_2 = find_points_2(pos_2, full_train_reps_2, R1, k1)

        ######

        train_fps = []

        for m, i in zip(train_mols, train[train_inds]):

            r1 = make_representation_morfeus_var_gpt(total_beads_1, m[0], 0, R2, k2, rep_=full_reps_1[i])
            r2 = make_representation_morfeus_var_gpt(total_beads_2, m[1], 0, R2, k2, rep_=full_reps_2[i])

            train_fps.append(np.hstack((r1, r2 , prop_descs[i])))


        val_fps = []

        for m, i in zip(val_mols, train[val_inds]):
            r1 = make_representation_morfeus_var_gpt(total_beads_1, m[0], 0, R2, k2, rep_=full_reps_1[i])
            r2 = make_representation_morfeus_var_gpt(total_beads_2, m[1], 0, R2, k2, rep_=full_reps_2[i])

            val_fps.append(np.hstack((r1, r2, prop_descs[i])))

        test_fps = []

        for m, i in zip(test_mols, test):
            r1 = make_representation_morfeus_var_gpt(total_beads_1, m[0], 0, R2, k2, rep_=full_reps_1[i])
            r2 = make_representation_morfeus_var_gpt(total_beads_2, m[1], 0, R2, k2, rep_=full_reps_2[i])

            test_fps.append(np.hstack((r1, r2, prop_descs[i])))

        ######

        train_fps = np.array(train_fps)

        test_fps = np.array(test_fps)
        val_fps = np.array(val_fps)

        scaler = StandardScaler()

        scaler.fit(train_fps)

        train_descs = scaler.transform(train_fps)
        val_descs = scaler.transform(val_fps)
        test_descs = scaler.transform(test_fps)

        '''pca = PCA(n_components=len(train_fps[0]))

        pca.fit(train_fps)

        train_descs = scaler.transform(train_descs)
        val_descs = scaler.transform(val_descs)
        test_descs = scaler.transform(test_descs)'''

        test_pred, train_pred,val_pred = train_SVC(train_descs, test_descs, val_descs, train_y,
                                                        val_y)

        test_ps[test] = test_pred

    ###measure val RMSE and prs

    return test_ps


def bootstrap_hyper(mols,y_data,full_reps_1,full_reps_2,R1,R2,k1,k2,model,prop_descs, hyperparam_combs):

    kf = KFold(n_splits=10, shuffle=True)

    val_ps =[]

    for c in range(len(hyperparam_combs)):

        val_ps.append(np.zeros( len(y_data)))

    for train, val in kf.split(mols):

        train_mols = mols[train]
        val_mols = mols[val]

        train_y = y_data[train]

        full_train_reps_1 = [full_reps_1[i] for i in train]
        full_train_reps_2 = [full_reps_2[i] for i in train]

        #####

        pos_1 = []
        pos_2 = []

        for m in train_mols:
            pos_1.append(m[0].GetConformer().GetPositions())
            pos_2.append(m[1].GetConformer().GetPositions())

        total_beads_1 = find_points_3(pos_1, full_train_reps_1, R1, k1,train_y,0.05,1)
        total_beads_2 = find_points_3(pos_2, full_train_reps_2, R1, k1,train_y,0.05,1)

        ######

        train_fps = []

        for m, i in zip(train_mols, train):
            r1 = make_representation_morfeus_var_gpt(total_beads_1, m[0], 0, R2, k2, rep_=full_reps_1[i])
            r2 = make_representation_morfeus_var_gpt(total_beads_2, m[1], 0, R2, k2, rep_=full_reps_2[i])

            train_fps.append(np.hstack((r1, r2 , prop_descs[i])))

        val_fps = []

        for m, i in zip(val_mols, val):
            r1 = make_representation_morfeus_var_gpt(total_beads_1, m[0], 0, R2, k2, rep_=full_reps_1[i])
            r2 = make_representation_morfeus_var_gpt(total_beads_2, m[1], 0, R2, k2, rep_=full_reps_2[i])

            val_fps.append(np.hstack((r1, r2, prop_descs[i])))

        ######

        scaler = StandardScaler()

        scaler.fit(train_fps)

        train_descs = scaler.transform(train_fps)
        val_descs = scaler.transform(val_fps)

        mean_y = np.mean(train_y)
        std_y = np.std(train_y)

        train_y_data = (train_y - mean_y) / std_y

        for k,v in zip( hyperparam_combs.keys() , hyperparam_combs.values()):

            if model == "GB":

                val_pred, model_ = train_GBR_hyper(train_descs, val_descs, train_y_data,v)
            elif model == "EN":

                val_pred, model_ = train_EN_hyper(train_descs, val_descs, train_y_data,v)

            val_pred = val_pred * std_y + mean_y

            val_ps[int(k)][val] = val_pred

    return val_ps



class Settings:
    def __init__(self, ):
        self.InputFile1 = ''
        self.InputFile2 = ''

        self.OutputFolder = ''
        self.FittedObjects = ''

        self.Scaler = os.path.dirname(os.path.realpath(sys.argv[0])) + '/derivation_utils/R_7.4_kde_scaler.p'
        self.NCores = os.cpu_count() - 1

        self.NConf = False
        self.RConf = False


class Results:

    def __init__(self, ):
        self.Reps = []


if __name__ == '__main__':

    def run_n(mols, y_data, model, R1, R2, k1, k2,n_bootstraps):

        test_ps = np.zeros((n_bootstraps, len(y_data)))

        full_reps_1 = []
        full_reps_2 = []

        prop_descs = []

        for m in tqdm.tqdm(mols):

            print(m[0].GetProp("TRB_ID") , m[1].GetProp("TRB_ID"))

            full_reps_1.append(make_representation_2(m[0], 0))
            full_reps_2.append(make_representation_2(m[1], 0))
            prop_descs.append(descriptor_gs(m[0]) )

        pickle.dump(full_reps_1 , open("r1.p","wb"))
        pickle.dump(full_reps_2 , open("r2.p","wb"))

        full_reps_1 = pickle.load( open("r1.p", "rb"))
        full_reps_2 = pickle.load( open("r2.p", "rb"))

        args = []

        for n in range(n_bootstraps):
            args.append([mols,y_data,full_reps_1, full_reps_2, R1, R2, k1, k2, model,prop_descs])

        with mp.Pool(5) as p, tqdm.tqdm(total=len(args)) as pbar:
            res = [p.apply_async(
                bootstrap, args=args[i], callback=lambda _: pbar.update(1)) for i in range(len(args))]
            res_vector = [r.get() for r in res]

        for n, r in enumerate(res_vector):
            test_ps[n] = r
            # val_ps[n] = res_vector[0]

        test_ps_ = np.mean(test_ps, axis=0)
        test_stds = np.std(test_ps, axis=0)

        test_vs = y_data

        return test_vs, test_ps_, test_stds

    def run_hyperParams(mols, y_data, model, R1, R2, k1, k2,n_bootstraps):

        full_reps_1 = []
        full_reps_2 = []
        prop_descs = []

        for m in tqdm.tqdm(mols):

            full_reps_1.append(make_representation(m[0], 0))
            full_reps_2.append(make_representation(m[1], 0))
            prop_descs.append(descriptor_gs(m[0]) )

        combs = {}
        cb = 0

        if model == "GB":

            max_depth_ = [None, 1, 3, 5]
            n_est_ = [200,500]
            lr_ = [0.01, 0.05]

            for lr in lr_:

                for n_est in n_est_:

                    for max_depth in max_depth_:

                        hyper_params = { 'lr' : lr  , 'n_est' : n_est , 'max_depth' : max_depth}
                        combs[str(cb)] = hyper_params
                        cb+=1

        elif model == "EN":

            alpha = [0.05, 0.1]
            l1_ratio = [0.1, 0.5, 0.75]
            for alpha_ in alpha:

                for l1_ratio_  in l1_ratio:


                    hyper_params = {'alpha': alpha_, 'l1_ratio': l1_ratio_ }
                    combs[str(cb)] = hyper_params
                    cb += 1

        else:

            quit()

        args = []

        for n in range(n_bootstraps):
            args.append([mols,y_data,full_reps_1, full_reps_2, R1, R2, k1, k2, model,prop_descs , combs])

        with mp.Pool(5) as p, tqdm.tqdm(total=len(args)) as pbar:
            res = [p.apply_async(
                bootstrap_hyper, args=args[i], callback=lambda _: pbar.update(1)) for i in range(len(args))]
            res_vector = [r.get() for r in res]

        val_out = [[] for i in range(len(combs))]

        for bootstrap, r in enumerate(res_vector):

            for cbn,comb_r in enumerate(r):

                val_out[cbn].append(comb_r)

        for c,comb_r in enumerate(val_out):

            test_ps_ = np.mean(comb_r, axis=0)

            psr = pearsonr(y_data, test_ps_)

            combs[str(c)]['pearsons'] = psr

        return combs

    print("=================================================")
    print("# Fuzzy Pharmacophore Molecular Similarity      #")
    print("# Alex Howarth, TBio                            #")
    print("=================================================\n\n")

    settings = Settings()

    settings.InputFile1 = "/Users/alexanderhowarth/Desktop/Projects/YTHDC1/FP4.1_models/53551/HTRF_cac594dd-5549-4114-86fa-385b4603ef87_update.sdf"
    settings.InputFile2 = "/Users/alexanderhowarth/Desktop/Projects/YTHDC1/FP4.1_models/53179/HTRF_7a0dfad3-163b-4de2-9d4c-0ae275768599_update.sdf"

    R1 = 2
    R2 = 4
    k1 = 2
    k2 = 1

    dict_ = {}

    sol_dict = {}


    for m in tqdm.tqdm(Chem.SDMolSupplier("/Users/alexanderhowarth/Desktop/Projects/YTHDC1/Solubility/v1903_solubility.sdf")):

        try:
            sol_dict[m.GetProp("ID")] = m.GetProp("Solubility [µM]")

        except:

            None

    for m in tqdm.tqdm(Chem.SDMolSupplier(settings.InputFile1, removeHs=False)):

        #y = float(m.GetProp("HTRF Absolute IC50 Mean"))

        try:

            y =float( sol_dict[ m.GetProp("TRB_ID")]) > 1.7

            dict_[m.GetProp("TRB_ID")] = [[ m ] , y  ]

        except:

            None


    for m in tqdm.tqdm(Chem.SDMolSupplier(settings.InputFile2, removeHs=False)):


        try:
            ID = m.GetProp("TRB_ID")

            dict_[ID][0].append(m)

        except:

            None


    mols =[]
    y_data =[ ]
    scores = []

    for k, v in zip(dict_.keys() , dict_.values()):

        mols.append(v[0])

        scores.append(-float(v[0][0].GetProp("Score")))
        y_data.append(v[1])

    mols = np.array(mols)
    y_data = np.array(y_data)

    ###### run an evaluation round

    n_bootstraps = 10

    color = "C4"

    av_vs, av_ps,stds  = run_n(mols,y_data,"XB",R1,R2,k1,k2, n_bootstraps)


    fpr, tpr, thresholds = metrics.roc_curve(av_vs, av_ps, pos_label=1)

    auc_ = metrics.auc(fpr, tpr)

    print(auc_)

    print(av_ps)

    plt.plot(fpr,tpr)

    plt.show()
    quit()

    '''scaler = StandardScaler()

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