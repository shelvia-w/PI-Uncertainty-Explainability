import numpy as np
import tensorflow as tf
from scipy import stats
import os
import pickle

from sklearn.ensemble import RandomForestClassifier

from src.models import get_optimizer, mlp
from src.datasets import prefetch_dataset

def sample_from_sphere(d):
    vec = np.random.randn(d, 1)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

def psi_bin_train(x, y, n_projs, n_bins):
    y = np.argmax(y, axis=1)
    psi_bin_data = {}
    all_hist = []
    all_bin_edges = []
    all_thetas = []
    for _ in range(n_projs):
        theta = sample_from_sphere(x.shape[1])
        all_thetas.append(theta)
        thetax = np.dot(x,theta)
        hist_list = []
        bin_edges_list = []
        for k in range(np.max(y)+1):
            idx = np.where(y == k)[0]
            thetax_class = thetax[idx]
            hist, bin_edges = np.histogram(thetax_class, bins=n_bins, density=False)
            hist_list.append(hist)
            bin_edges_list.append(bin_edges)
        all_hist.append(hist_list)
        all_bin_edges.append(bin_edges_list)
    n_class_list = []
    for k in range(np.max(y)+1):
        idx = np.where(y == k)[0]
        n_class_list.append(len(idx))
    #shape: (n_projs, n_class, n_bins)
    psi_bin_data['hist'] = np.array(all_hist)
    #shape: (n_projs, n_class, n_bins+1)
    psi_bin_data['bin_edges'] = np.array(all_bin_edges)
    #shape: (n_projs,dim_x)
    psi_bin_data['thetas'] = np.squeeze(all_thetas)
    psi_bin_data['n_classes'] = np.max(y)+1
    psi_bin_data['n_train'] = len(x)
    psi_bin_data['class_prob'] = np.array(n_class_list)/len(x)
    return psi_bin_data

def psi_bin_val(x, y, psi_bin_data, n_projs):
    pmi_list = []
    n_bins = psi_bin_data['hist'].shape[2]
    n_class_list = np.array(psi_bin_data['class_prob'] * psi_bin_data['n_train']).astype(int)
    thetax = np.dot(x, psi_bin_data['thetas'][:n_projs].T)   # shape: (n_samples, n_projs)
    for m in range(n_projs):
        p_thetax_given_y = []
        for k in range(psi_bin_data['n_classes']):
            bin_idx = np.clip(np.digitize(thetax[:,m], psi_bin_data['bin_edges'][m][k]), 1, n_bins)
            p = psi_bin_data['hist'][m][k][bin_idx - 1] / n_class_list[k]
            p_thetax_given_y.append(p)
        p_thetax_given_y = np.array(p_thetax_given_y)
        p_thetax = np.sum(psi_bin_data['class_prob'][:, np.newaxis] * p_thetax_given_y, axis=0)
        num = np.sum(y * np.array(p_thetax_given_y).T, axis=1)
        den = np.array([p_thetax])
        pmi = np.squeeze(np.log2(np.clip(num, 1e-5, None))-np.log2(np.clip(den, 1e-5, None)))
        pmi_list.append(pmi)
    pmi_arr = np.array(pmi_list).T # shape: (n_samples, n_projs)
    psi_mean = np.mean(pmi_arr, axis=1)
    return psi_mean, pmi_arr

def psi_gauss_train(x, y, n_projs):
    y = np.argmax(y, axis=1)
    psi_gauss_data = {}
    all_mu = []
    all_std = []
    all_thetas = []
    for _ in range(n_projs):
        theta = sample_from_sphere(x.shape[1])
        all_thetas.append(theta)
        thetax = np.dot(x,theta)
        mu_list = []
        std_list = []
        for k in range(np.max(y)+1):
            idx = np.where(y == k)[0]
            thetax_class = thetax[idx]
            mu, std = stats.norm.fit(thetax_class)
            mu_list.append(mu)
            std_list.append(std)
        all_mu.append(mu_list)
        all_std.append(std_list)
    n_class_list = []
    for k in range(np.max(y)+1):
        idx = np.where(y == k)[0]
        n_class_list.append(len(idx))
    #shape: (n_projs, n_class)
    psi_gauss_data['mu'] = np.array(all_mu)
    #shape: (n_projs, n_class)
    psi_gauss_data['std'] = np.array(all_std)
    #shape: (n_projs, dim_x)
    psi_gauss_data['thetas'] = np.squeeze(all_thetas)
    psi_gauss_data['n_classes'] = np.max(y)+1
    psi_gauss_data['class_prob'] = np.array(n_class_list)/len(x)
    return psi_gauss_data

def psi_gauss_val(x, y, psi_gauss_data, n_projs):
    pmi_list = []
    thetax = np.dot(x, psi_gauss_data['thetas'][:n_projs].T)   # shape: (n_samples, n_projs)
    for m in range(n_projs):
        p_thetax_given_y = []
        for k in range(psi_gauss_data['n_classes']):
            p = stats.norm.pdf(thetax[:,m], psi_gauss_data['mu'][m][k], psi_gauss_data['std'][m][k])
            p_thetax_given_y.append(p)
        p_thetax_given_y = np.array(p_thetax_given_y)
        p_thetax = np.sum(psi_gauss_data['class_prob'][:, np.newaxis] * p_thetax_given_y, axis=0)
        num = np.sum(y * np.array(p_thetax_given_y).T, axis=1)
        den = np.array([p_thetax])
        pmi = np.squeeze(np.log2(np.clip(num, 1e-5, None))-np.log2(np.clip(den, 1e-5, None)))
        pmi_list.append(pmi)
    pmi_arr = np.array(pmi_list).T # shape: (n_samples, n_projs)
    psi_mean = np.mean(pmi_arr, axis=1)
    return psi_mean, pmi_arr

def psi_rf_train(x, y, n_projs, save_path):
    if not os.path.exists(f'{save_path}/psi_models'):
        print("Making directory", f'{save_path}/psi_models')
        os.makedirs(f'{save_path}/psi_models')
    
    all_thetas = []
    for proj in range(n_projs):
        theta = sample_from_sphere(x.shape[1])
        all_thetas.append(theta)
        thetax = np.dot(x,theta)
        clf = RandomForestClassifier(max_depth=2, random_state=0)
        clf.fit(thetax, y)
        with open(f'{save_path}/psi_models/psi_model_{proj+1}','wb') as f:
            pickle.dump(clf,f)
    np.save(f'{save_path}/all_thetas.npy', np.array(all_thetas))
    
def psi_rf_val(x, k, thetas, class_prob, n_projs, save_path):
    pmi_list = []
    for proj in range(n_projs):
        thetax = np.dot(x,thetas[proj])
        with open(f'{save_path}/psi_models/psi_model_{proj+1}','rb') as f:
            clf = pickle.load(f)
        pred_prob = clf.predict_proba(thetax)
        p_y_given_thetax = np.clip(pred_prob[:,k], 1e-5, None)
        p_y = class_prob[k]
        pmi = np.log2(p_y_given_thetax/p_y)
        pmi_list.append(pmi)
    pmi_arr = np.array(pmi_list).T # shape: (n_samples, n_projs)
    psi_mean = np.mean(pmi_arr, axis=1)
    return psi_mean, pmi_arr  

def psi_neural_train(ds, cfg, n_projs, save_path):
    if not os.path.exists(f'{save_path}/psi_models'):
        print("Making directory", f'{save_path}/psi_models')
        os.makedirs(f'{save_path}/psi_models')
    
    all_thetas = []
    for proj in range(n_projs):
        theta = np.float32(sample_from_sphere(ds.element_spec[0].shape[0]))
        all_thetas.append(theta)
        ds_theta = ds.map(lambda x, y: (tf.tensordot(x, theta, axes=1), y))
        ds_theta = prefetch_dataset(ds_theta, batch_size=cfg['batch_size'])
        model = mlp(ds_theta, n_layers=1, n_hidden=128)
        optimizer = get_optimizer(cfg['optimizer'], learning_rate=0.01)
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        model.compile(loss=loss_fn, optimizer=optimizer, metrics=['accuracy'])
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        model.fit(ds_theta, verbose=1, epochs=100, callbacks=[early_stop,])
        model.save(f'{save_path}/psi_models/psi_model_{proj+1}.keras')
                   
    np.save(f'{save_path}/all_thetas.npy', np.array(all_thetas))
                   
def psi_neural_val(ds, thetas, class_prob, cfg, n_projs, save_path):
    pmi_list = []
    for proj in range(n_projs): 
        model = tf.keras.models.load_model(f'{save_path}/psi_models/psi_model_{proj+1}.keras')
        ds_theta = ds.map(lambda x, y: (tf.tensordot(x, thetas[proj], axes=1), y)).batch(cfg['batch_size'])           
        class_conditional_prob = np.max(model.predict(ds_theta), axis=1)
        
        y = np.argmax([y for x,y in ds], axis=1)                                   
        p_y = np.array([class_prob[label] for label in y])
        
        pmi = np.log2(class_conditional_prob/p_y)
        pmi_list.append(pmi)
    pmi_arr = np.array(pmi_list).T # shape: (n_samples, n_projs)
    psi_mean = np.mean(pmi_arr, axis=1)
    return psi_mean, pmi_arr
                                           
    
    