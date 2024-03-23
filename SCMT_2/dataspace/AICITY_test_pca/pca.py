import cv2
import numpy as np
import pickle
from os.path import join
from sklearn.decomposition import PCA, FastICA
import matplotlib.pyplot as plt
import time
from sklearn.metrics import  pairwise 
from sklearn.neighbors.nearest_centroid import NearestCentroid
import scipy.stats
from tqdm import tqdm

from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.covariance import ShrunkCovariance, LedoitWolf
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def compute_scores(X,n_components):
    pca = PCA(svd_solver="full")

    pca_scores = []
    for n in tqdm(n_components):
        pca.n_components = n
        pca_scores.append(np.mean(cross_val_score(pca, X)))

    return pca_scores

#n_components = np.arange(0.85, 0.96, 0.01)  # options for n_components

s_time = time.time()
sequences = ['c041','c042','c043','c044','c045','c046']
for i, seq in enumerate(sequences, start=1):
    print('processing the {}th video {}...'.format(i, seq))
    detection_file = join('./', seq + '.npy')
    if detection_file is not None:
        detections = np.load(detection_file)
    info = detections[:,:10]
    feat = detections[:,-2048:]

    mean_feat = feat - np.mean(feat,axis=0)   
    pca = PCA()    
    #pca = PCA(n_components=0.9,random_state=2)
    pca.fit(mean_feat)
    pca_feat = pca.transform(mean_feat)
    pca_feat = pca_feat/np.sqrt(np.sum(np.square(pca_feat),axis=1))[:,np.newaxis]
    
    new_feat = np.concatenate((info,pca_feat),axis=1)
  
    ##draw_pic
    #exp_var_pca = pca.explained_variance_ratio_
    #cum_sum_eigenvalues = np.cumsum(exp_var_pca)
    #r = np.arange(0.8, 1.00, 0.01)
    #components = []
    #print(np.max(cum_sum_eigenvalues))    
    #for ratio in r:
    #    a = np.where(cum_sum_eigenvalues>ratio)[0]
    #    components.append(a[0]+1)
    #print(r)
    #print(components)    
    ##plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
    #plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
    #plt.ylabel('Accumulated variance ratio')
    #plt.xlabel('Principal component index')
    #plt.legend(loc='best')
    #plt.tight_layout()
    #plt.savefig(f"SCT_PCA_Cumulative_{seq}.png")
    #plt.clf()
    ##exit()
    
    #pca_scores = compute_scores(pca_feat,components)
    #n_components_pca = components[np.argmax(pca_scores)]
    #
    #print("best n_components by PCA CV = %d" % n_components_pca)
    #print(r[np.argmax(pca_scores)])
    #print(pca_scores)
    
    #print(new_feat.shape)
    np.save('./pca_feat/{}.npy'.format(seq), new_feat)    


print(time.time()-s_time)    










