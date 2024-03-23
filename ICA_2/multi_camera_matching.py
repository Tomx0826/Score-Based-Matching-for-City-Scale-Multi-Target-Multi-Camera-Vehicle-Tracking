import os, sys, pdb
import numpy as np
import cv2
import pickle
import argparse
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
from rerank import re_ranking
from collections import defaultdict
import scipy.stats as st
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import  pairwise 
from sklearn.neighbors.nearest_centroid import NearestCentroid
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

common_feat = 150
# Those dicts are adopted to preserve all valid crossing vehicles. 
# For example: if one vehicle leaves port 1 at current camera, it should drive into port 1 at next camera.
in_valid_positive_directions  = ((1, 1), (1, 0), (1, 2))
out_valid_positive_directions = ((1, 1), (0, 1), (2, 1))
in_valid_negative_directions  = ((3, 3), (3, 0), (3, 2))
out_valid_negative_directions = ((3, 3), (0, 3), (2, 3))

# Those dicts are used to preserve the vehicles that must be in two successive cameras.
in_positive_direction_time_thre_dict  = {42: 550, 43: 180, 44: 440, 45: 240, 46: 360}
out_positive_direction_time_thre_dict = {41: 1111, 42: 1640, 43: 1610, 44: 1450, 45: 1610}
in_negative_direction_time_thre_dict  = {41: 600, 42: 350, 43: 560, 44: 290, 45: 760}
out_negative_direction_time_thre_dict = {42: 1240, 43: 1710, 44: 1520, 45: 1582, 46: 1430}

# The purpose of this dict is to preserve the vehicles whose travel time is possible between two ports.
# hard constraint for refine distance matrix
two_track_valid_pass_time_for_mask_dict = {(41, 42): [450, 1080], (42, 43): [130, 250], (43, 44): [340, 545], 
                                  (44, 45): [110, 635], (45, 46): [150, 900],
                                  (46, 45): [200, 730], (45, 44): [120, 700], (44, 43): [95, 1005], 
                                  (43, 42): [195, 530], (42, 41): [410, 570]}
# loose constraint for post-process
two_track_valid_pass_time_dict = {(41, 42): [300, 1500], (42, 43): [130, 800], (43, 44): [280, 680], 
                                  (44, 45): [50, 800], (45, 46): [80, 1000],
                                  (46, 45): [150, 850], (45, 44): [80, 800], (44, 43): [80, 1090], 
                                  (43, 42): [150, 600], (42, 41): [350, 850]}

# hyper parameters
args_params_dict = {(41, 42): {'topk': [0.3,1], 'r_rate': [0.01,0.3], 'k1': 13, 'k2': 5, 'lambda_value': 0.7, 
                     'alpha': 0.8, 'long_time_t': 500, 'short_time_t': 1000, 'use_ica': [1,1],'num_search_times': 2}, 
                    (42, 43): {'topk': [0.3], 'r_rate': [0.01], 'k1': 13, 'k2': 5, 'lambda_value': 0.7,  
                     'alpha': 0.8, 'long_time_t': 500, 'short_time_t': 1000, 'use_ica': [0,1],'num_search_times': 1},
                    (43, 44): {'topk': [0.4,1], 'r_rate': [0.01,0.3], 'k1': 13, 'k2': 8, 'lambda_value': 0.4,  
                     'alpha': 1.1, 'long_time_t': 500, 'short_time_t': 1000, 'use_ica': [1,1],'num_search_times': 2},
                    (44, 45): {'topk': [0.4], 'r_rate': [0.01], 'k1': 12, 'k2': 7, 'lambda_value': 0.6, 
                     'alpha': 1.1, 'long_time_t': 1000, 'short_time_t': 1000, 'use_ica': [0,0],'num_search_times': 1},
                    (45, 46): {'topk': [0.3], 'r_rate':[0.001], 'k1': 13, 'k2': 5, 'lambda_value': 0.7,         
                     'alpha': 1.1, 'long_time_t': 500, 'short_time_t': 1000, 'use_ica': [0,0],'num_search_times': 1}, 
                    (46, 45): {'topk': [0.4,0.1], 'r_rate': [0.05, 0.5], 'k1': 13, 'k2': 5, 'lambda_value': 0.7, 
                     'alpha': 0.8, 'long_time_t': 1000, 'short_time_t': 1000, 'use_ica': [0,0],'num_search_times': 2},
                    (45, 44): {'topk': [0.8], 'r_rate': [0.01], 'k1': 12, 'k2': 7, 'lambda_value': 0.6, 
                     'alpha': 1.1, 'long_time_t': 1000, 'short_time_t': 1000, 'use_ica': [0,1],'num_search_times': 1},
                    (44, 43): {'topk': [0.4], 'r_rate': [0.01], 'k1': 13, 'k2': 5, 'lambda_value': 0.7,  
                     'alpha': 1.1, 'long_time_t': 500, 'short_time_t': 1000, 'use_ica': [0,0],'num_search_times': 1},
                    (43, 42): {'topk': [0.6], 'r_rate': [0.001], 'k1': 13, 'k2': 5, 'lambda_value': 0.7, 
                     'alpha': 1.1, 'long_time_t': 500, 'short_time_t': 1000, 'use_ica': [0,0],'num_search_times': 1},
                    (42, 41): {'topk': [0.4], 'r_rate': [0.01], 'k1': 12, 'k2': 7, 'lambda_value': 0.6, 
                     'alpha': 1.1, 'long_time_t': 1000, 'short_time_t': 1000, 'use_ica': [0,0],'num_search_times': 1},} 

              
def argument_parser():
    """ Argument parser
    Receive args

    Args:
        None
        
    Returns:
        parser: An argument object that contains all args

    Raises:
        None
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--src_root', type=str, default="data/preprocessed_data/", 
            help='the root path of tracked files of single camera with submission format')
    parser.add_argument('--dst_root', type=str, default="submit/", 
            help='the root path of the generated file to submit')
    parser.add_argument('--mode', type=str, default='linear')
    parser.add_argument('--st_dim', type=int, default=0)
    parser.add_argument('--en_dim', type=int, default=2048)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--r_rate', type=float, default=0.5)

    parser.add_argument('--k1', type=int, default=12)
    parser.add_argument('--k2', type=int, default=7)
    parser.add_argument('--lambda_value', type=float, default=0.6)

    parser.add_argument('--alpha', type=float, default=1.1)
    parser.add_argument('--long_time_t', type=float, default=1000)
    parser.add_argument('--short_time_t', type=float, default=1000)

    parser.add_argument('--num_search_times', type=int, default=1)

    parser.add_argument('--occ_rate', type=float, default=1.0)
    parser.add_argument('--occ_alpha', type=float, default=0.)

    return parser

class MultiCameraMatching(object):
    """ This class is used to match tracklets among all different cameras.

    Attributes:
        cam_arr: camera id array
        track_arr: tracklet array
        in_dir_arr: the "in zone" in our paper
        out_dir_arr: the "out zone" in our paper
        in_time_arr: the time when a tracklet enter the "in zone"
        out_time_arr: the time when a tracklet exit the "out zone"
        feat_dict: the tracklet features
        feat_arr: it has been deprecated
    """

    def __init__(self, cam_arr, track_arr, in_dir_arr, out_dir_arr, 
            in_time_arr, out_time_arr, feat_dict, track_dict,
            topk=5, r_rate=0.5, 
            k1=12, k2=7, lambda_value=0.6,
            alpha=1.1, long_time_t=1000, short_time_t=1000,
            num_search_times=1, fr_dict=None,
            trun_dict=None, occ_rate=0.5, occ_alpha=0.):
        self.cam_arr = cam_arr
        self.track_arr = track_arr
        self.in_dir_arr = in_dir_arr
        self.out_dir_arr = out_dir_arr
        self.in_time_arr = in_time_arr
        self.out_time_arr = out_time_arr

        self.feat_dict = feat_dict
        self.fr_dict = fr_dict
        self.trun_dict = trun_dict
        self.track_dict = track_dict
        ### params
        self.topk = topk
        self.r_rate = r_rate

        self.k1 = k1
        self.k2 = k2
        self.lambda_value = lambda_value

        self.alpha = alpha
        self.long_time_t = long_time_t
        self.short_time_t = short_time_t
        self.use_ica = 0
        self.num_search_times = num_search_times

        self.occ_rate = occ_rate
        self.occ_alpha = occ_alpha
        
        self.pca1 = None
        self.pca2 = None

        self.occ_out = None
        self.occ_in = None
        self.start_time = 0

        self.index_map_out = None
        self.index_map_in = None
        
        self.global_id_arr = np.zeros_like(cam_arr) - 1

    def select_map_arr(self, cam_id, is_out, direction=True):
        """ Select valid vehicles...
        Args:
            cam_id: camera id
            is_out: The track is out of the camera, else in
            direction: True is positive and False is negative
        """
        map_arr = (self.cam_arr == cam_id)
        dir_map = np.zeros_like(map_arr)
        if is_out: # 保留正确的进/出口车辆
            valid_directions = out_valid_positive_directions if direction else out_valid_negative_directions
            for i in range(len(valid_directions)):
                tmp_map = (self.in_dir_arr == valid_directions[i][0]) & (self.out_dir_arr == valid_directions[i][1])
                dir_map |= tmp_map
            tmp_map = (self.in_dir_arr == -1) & (self.out_dir_arr == valid_directions[0][1]) # for those tracks at the beginning
            dir_map |= tmp_map
            if cam_id == 46: # for the special case!
                tmp_map = (self.in_dir_arr == 1) & (self.out_dir_arr == 3) # for the special case in camera 46
                dir_map |= tmp_map

            t_thre = out_positive_direction_time_thre_dict[cam_id] if direction \
                    else out_negative_direction_time_thre_dict[cam_id] # time threshold
            tmp_map = (self.out_dir_arr == valid_directions[0][1]) & (self.out_time_arr < t_thre)
            dir_map &= tmp_map
        else:
            valid_directions = in_valid_positive_directions if direction else in_valid_negative_directions
            for i in range(len(valid_directions)):
                tmp_map = (self.in_dir_arr == valid_directions[i][0]) & (self.out_dir_arr == valid_directions[i][1])
                dir_map |= tmp_map
            tmp_map = (self.in_dir_arr == valid_directions[0][0]) & (self.out_dir_arr == -1) # for those tracks in the last time
            dir_map |= tmp_map
            if cam_id == 46: # for the special case!
                tmp_map = (self.in_dir_arr == 1) & (self.out_dir_arr == 3) # for the special case in camera 46
                dir_map |= tmp_map

            t_thre = in_positive_direction_time_thre_dict[cam_id] if direction \
                    else in_negative_direction_time_thre_dict[cam_id] # time threshold
            tmp_map = (self.in_dir_arr == valid_directions[0][0]) & (self.in_time_arr > t_thre)
            dir_map &= tmp_map

        map_arr &= dir_map
        return map_arr

    def select_map_arr_interval(self, cam_id, is_out, interval=[0, 2001], direction=True):
        """ Select valid vehicles...
        Args:
            cam_id: camera id
            is_out: The track is out of the camera, else in
            direction: True is positive and False is negative
        """
        
        map_arr = (self.cam_arr == cam_id)
        dir_map = np.zeros_like(map_arr) 
        if is_out: # 保留正确的进/出口车辆
            valid_directions = out_valid_positive_directions if direction else out_valid_negative_directions
            for i in range(len(valid_directions)):
                tmp_map = (self.in_dir_arr == valid_directions[i][0]) & (self.out_dir_arr == valid_directions[i][1])
                dir_map |= tmp_map
            tmp_map = (self.in_dir_arr == -1) & (self.out_dir_arr == valid_directions[0][1]) # for those tracks at the beginning
            dir_map |= tmp_map
            if cam_id == 46 or cam_id == 43: # for the special case!
                tmp_map = (self.in_dir_arr == 1) & (self.out_dir_arr == 3) # for the special case in camera 46
                dir_map |= tmp_map

            t_min_thre, t_max_thre = interval
            tmp_map = (self.out_dir_arr == valid_directions[0][1]) & (t_min_thre < self.out_time_arr) & \
                    (self.out_time_arr < t_max_thre)
            dir_map &= tmp_map
        else:
            valid_directions = in_valid_positive_directions if direction else in_valid_negative_directions
            for i in range(len(valid_directions)):
                tmp_map = (self.in_dir_arr == valid_directions[i][0]) & (self.out_dir_arr == valid_directions[i][1])
                dir_map |= tmp_map
            tmp_map = (self.in_dir_arr == valid_directions[0][0]) & (self.out_dir_arr == -1) # for those tracks in the last time
            dir_map |= tmp_map
            if cam_id == 46 or cam_id == 43: # for the special case!
                tmp_map = (self.in_dir_arr == 1) & (self.out_dir_arr == 3) # for the special case in camera 46
                dir_map |= tmp_map

            t_min_thre, t_max_thre = interval
            tmp_map = (self.in_dir_arr == valid_directions[0][0]) & (t_min_thre < self.in_time_arr) & \
                    (self.in_time_arr < t_max_thre) 
            dir_map &= tmp_map

        map_arr &= dir_map
        return map_arr

    def do_matching(self, cam_out_arr, cam_in_arr, track_out_arr, track_in_arr, 
            out_time_out_arr, in_time_in_arr, cam_out_id, cam_in_id, st_dim=0, en_dim=2048, Iter=0):
        n_out = cam_out_arr.shape[0]
        cam_out_feat_list = []
        track_out_feat_list = []
        index_out_list = []
        feat_out_list = []
        trun_out_list = []
        for i in range(n_out):
            f_out = np.array(self.feat_dict[cam_out_arr[i]][track_out_arr[i]])[:, st_dim:en_dim]
            trun_out = np.array(self.trun_dict[cam_out_arr[i]][track_out_arr[i]])
            index_out_list.append(np.ones(f_out.shape[0], dtype=np.int64) * i)
            feat_out_list.append(f_out)
            trun_out_list.append(trun_out)
        index_out_arr = np.concatenate(index_out_list)
        feat_out_arr = np.concatenate(feat_out_list) # nxc
        trun_out_arr = np.concatenate(trun_out_list) # n
        self.occ_out = trun_out_arr
        print ('done for preparing feat_out_arr')

        n_in = cam_in_arr.shape[0]
        cam_in_feat_list = []
        track_in_feat_list = []
        index_in_list = []
        feat_in_list = []
        trun_in_list = []
        for j in range(n_in):
            f_in = np.array(self.feat_dict[cam_in_arr[j]][track_in_arr[j]])[:, st_dim:en_dim]
            trun_in = np.array(self.trun_dict[cam_in_arr[j]][track_in_arr[j]])
            index_in_list.append(np.ones(f_in.shape[0], dtype=np.int64) * j)
            feat_in_list.append(f_in)
            trun_in_list.append(trun_in)

        index_in_arr = np.concatenate(index_in_list)
        feat_in_arr = np.concatenate(feat_in_list) # mxc
        trun_in_arr = np.concatenate(trun_in_list) # m
        self.occ_in = trun_in_arr
        print ('done for preparing feat_in_arr')

        print ('start to compute distance matrix...')
        self.start_time = time.time()
        matched_i, matched_j = self.compute_distance_matrix(feat_out_arr, feat_in_arr, index_out_arr, index_in_arr,
                    cam_out_arr, cam_in_arr, out_time_out_arr, in_time_in_arr, cam_out_id, cam_in_id, 
                    trun_out_arr, trun_in_arr, Iter) # nxm

        return matched_i, matched_j

    def compute_distance_matrix(self, feat_out_arr, feat_in_arr, index_out_arr, index_in_arr,
                    cam_out_arr, cam_in_arr, out_time_out_arr, in_time_in_arr, cam_out_id, cam_in_id,
                    trun_out_arr, trun_in_arr, Iter):

        # PCA
        use_ica = self.use_ica
        print("Start separate_pca/ica")
        time_extraction = time.time()
        anova_feat_out_arr, anova_feat_in_arr = self.separate_pca(feat_out_arr, feat_in_arr, index_out_arr, index_in_arr, cam_out_id, cam_in_id, use_ica[0], Iter, "1")
        #print(anova_feat_out_arr.shape)
        anova_feat_out_arr2, anova_feat_in_arr2 = self.separate_pca(feat_in_arr, feat_out_arr, index_in_arr, index_out_arr, cam_out_id, cam_in_id, use_ica[1], Iter, "2")
        #print(anova_feat_out_arr2.shape)
        print("Separate_pca/ica Done")
        print(time.time()-time_extraction)
        
        # rerank
        dist_mat = self.calculate_dist(anova_feat_out_arr,anova_feat_in_arr)
        dist_mat2 = self.calculate_dist(anova_feat_out_arr2, anova_feat_in_arr2)
        dist_mat2 = dist_mat2.T
        
        # mask with intervals
        tth_min, tth_max = two_track_valid_pass_time_for_mask_dict[(cam_out_id, cam_in_id)]
        out_time_out_box_arr = out_time_out_arr[index_out_arr] # n
        in_time_in_box_arr = in_time_in_arr[index_in_arr] # m
        n_out_box = out_time_out_box_arr.shape[0]
        n_in_box = in_time_in_box_arr.shape[0]
        out_time_out_box_mat = np.expand_dims(out_time_out_box_arr, 1).repeat(n_in_box, 1) # nxm
        in_time_in_box_mat = np.expand_dims(in_time_in_box_arr, 0).repeat(n_out_box, 0) # nxm
        
        alpha = self.alpha # param need to be adapted
        long_time_t = self.long_time_t # param need to be adapted
        short_time_t = self.short_time_t # param need to be adapted
        travel_time_mat = in_time_in_box_mat - out_time_out_box_mat
        travel_time_mask = np.ones_like(travel_time_mat)
        too_short_pairs_indices = (travel_time_mat < tth_min)
        too_long_pairs_indices = (travel_time_mat > tth_max)
        travel_time_mask[too_short_pairs_indices] = np.exp(alpha * (tth_min - \
                                                travel_time_mat[too_short_pairs_indices]) / short_time_t)
        travel_time_mask[too_long_pairs_indices] = np.exp(alpha * (travel_time_mat[too_long_pairs_indices] \
                                                        - tth_max) / long_time_t)

        dist_mat *= travel_time_mask    
        dist_mat2 *= travel_time_mask    

        # mask with occlusion
        occ_rate = self.occ_rate
        occ_alpha = self.occ_alpha
        th = 0.5
        n_ratio = 0.5
        occ_ratio = 0


        trun_out_arr = np.expand_dims(trun_out_arr, 1).repeat(n_in_box, 1) # nxm
        trun_out_mask_arr = (trun_out_arr > occ_rate)
        trun_out_weight_arr = np.ones_like(trun_out_arr)
        trun_out_weight_arr[trun_out_mask_arr] = np.exp(occ_alpha * (1 + trun_out_arr[trun_out_mask_arr]))
        
        trun_in_arr = np.expand_dims(trun_in_arr, 0).repeat(n_out_box, 0) # nxm
        trun_in_mask_arr = (trun_in_arr > occ_rate)
        trun_in_weight_arr = np.ones_like(trun_in_arr)
        trun_in_weight_arr[trun_in_mask_arr] = np.exp(occ_alpha * (1 + trun_in_arr[trun_in_mask_arr]))
    

    
        dist_mat *= trun_out_weight_arr
        dist_mat *= trun_in_weight_arr
        dist_mat2 *= trun_out_weight_arr
        dist_mat2 *= trun_in_weight_arr
       
        matched_i, matched_j = self.calculate_score(dist_mat, dist_mat2, index_out_arr, index_in_arr, cam_out_id, cam_in_id, out_time_out_arr, in_time_in_arr, Iter)
        return matched_i, matched_j
        

    def nearest_pair(self, cen_dist):
        inf = 10e6
        for i in range(len(cen_dist)):
            cen_dist[i,i] = inf
        match = []
        for i in range(30):
            min_distance_pair = np.where(cen_dist==np.min(cen_dist))
            row, col = min_distance_pair[0], min_distance_pair[1]
            cen_dist[row[0],col[0]] = inf
            cen_dist[col[0],row[0]] = inf
            match.append((row[0],col[0]))
        print(match)

    def separate_pca(self, feat_out_arr, feat_in_arr, index_out_arr, index_in_arr, cam_out_id, cam_in_id, use_ica, Iter, name):
    
        print(name)
        if name=="2":
            tmp = self.occ_out
            self.occ_out = self.occ_in
            self.occ_in = tmp
            self.blur_in = tmp
            tmp = self.index_map_out
            self.index_map_out = self.index_map_in
            self.index_map_in = tmp
            
 

        if Iter ==0 :
            pca = PCA(n_components=0.9,random_state=2)
            anova_feat_out_arr = feat_out_arr - np.mean(feat_out_arr,axis=0)
            anova_feat_in_arr = feat_in_arr - np.mean(feat_in_arr,axis=0)
            print("Start PCA")
            time_pca = time.time()
            pca.fit(anova_feat_out_arr, index_out_arr)
            
            
            #if name=="1":
            #    intra_distance = []
            #    for label in set(index_out_arr):
            #        X0 = anova_feat_out_arr[np.where(index_out_arr==label)] 
            #        sim0_intra = pairwise.pairwise_distances(X0, metric='cosine')
            #        num = len(sim0_intra)
            #        if num==1:
            #            continue
            #        intra_distance.append(sim0_intra.sum()/(num*num-num))
            #        print(f'ori intra: {self.index_map_out[label]} {sim0_intra.sum()/(num*num-num)}')
            #        
            #    print(f'--Intra-- out_track_2048: max {max(intra_distance)}, min {min(intra_distance)}, average {sum(intra_distance)/len(intra_distance)}')
            #    intra_distance = []
            #    for label in set(index_in_arr):
            #        X0 = anova_feat_in_arr[np.where(index_in_arr==label)] 
            #        sim0_intra = pairwise.pairwise_distances(X0, metric='cosine')
            #        num = len(sim0_intra)
            #        if num==1:
            #            continue
            #        intra_distance.append(sim0_intra.sum()/(num*num-num))
            #    print(f'--Intra-- in_track_2048: max {max(intra_distance)}, min {min(intra_distance)}, average {sum(intra_distance)/len(intra_distance)}')
            #    
            #    inter_distance = []
            #    clf = NearestCentroid()
            #    clf.fit(anova_feat_out_arr, index_out_arr)
            #    cen_dist = pairwise.pairwise_distances(clf.centroids_, metric='cosine')
            #    num = len(cen_dist)
            #    avrage_inter_disance = cen_dist.sum()/(num*num-num)
            #    max_inter_distance = np.max(cen_dist)
            #    for i in range(len(cen_dist)):
            #        cen_dist[i,i] = 10e6
            #    min_inter_distance = np.min(cen_dist)
            #    print(f'--Inter-- out_track_2048: max {max_inter_distance}, min {min_inter_distance}, average {avrage_inter_disance}')
            #    
            #    self.nearest_pair(cen_dist)
            #    inter_distance = []
            #    clf = NearestCentroid()
            #    clf.fit(anova_feat_in_arr, index_in_arr)
            #    cen_dist = pairwise.pairwise_distances(clf.centroids_, metric='cosine')
            #    num = len(cen_dist)
            #    avrage_inter_disance = cen_dist.sum()/(num*num-num)
            #    max_inter_distance = np.max(cen_dist)
            #    for i in range(len(cen_dist)):
            #        cen_dist[i,i] = 10e6
            #    min_inter_distance = np.min(cen_dist)
            #    print(f'--Inter-- in_track_2048: max {max_inter_distance}, min {min_inter_distance}, average {avrage_inter_disance}')   

            
            anova_feat_out_arr = pca.transform(anova_feat_out_arr)
            anova_feat_in_arr = pca.transform(anova_feat_in_arr)

            anova_feat_out_arr_pca_inver = pca.inverse_transform(anova_feat_out_arr)
            anova_feat_in_arr_pca_inver = pca.inverse_transform(anova_feat_in_arr)
            anova_feat_out_arr_pca_inver = anova_feat_out_arr_pca_inver/np.sqrt(np.sum(np.square(anova_feat_out_arr_pca_inver),axis=1))[:,np.newaxis]
            anova_feat_in_arr_pca_inver = anova_feat_in_arr_pca_inver/np.sqrt(np.sum(np.square(anova_feat_in_arr_pca_inver),axis=1))[:,np.newaxis]
            anova_feat_out_arr_pca_inver = anova_feat_out_arr_pca_inver - np.mean(anova_feat_out_arr_pca_inver,axis=0)
            anova_feat_in_arr_pca_inver = anova_feat_in_arr_pca_inver - np.mean(anova_feat_in_arr_pca_inver,axis=0)            
            print("PCA Done", time.time()-time_pca)
            
            
            n = len(list(set(index_out_arr))) + common_feat
            if use_ica ==1:
                ica_start_time = time.time()
                ica = FastICA(n_components=n,whiten='arbitrary-variance',max_iter=10,random_state=2)
                ica.fit(anova_feat_out_arr_pca_inver, index_out_arr)
                print(time.time()-ica_start_time)
                print("ICA done")
                anova_feat_out_arr_ica = ica.transform(anova_feat_out_arr_pca_inver)
                anova_feat_in_arr_ica = ica.transform(anova_feat_in_arr_pca_inver)    
                
                anova_feat_out_arr_ica_inver = ica.inverse_transform(anova_feat_out_arr_ica)
                anova_feat_in_arr_ica_inver = ica.inverse_transform(anova_feat_in_arr_ica)
                anova_feat_out_arr_ica_inver = anova_feat_out_arr_ica_inver/np.sqrt(np.sum(np.square(anova_feat_out_arr_ica_inver),axis=1))[:,np.newaxis]
                anova_feat_in_arr_ica_inver = anova_feat_in_arr_ica_inver/np.sqrt(np.sum(np.square(anova_feat_in_arr_ica_inver),axis=1))[:,np.newaxis]                
                
                print("ica dim:",anova_feat_out_arr_ica.shape[1])
                anova_feat_out_arr = anova_feat_out_arr/np.sqrt(np.sum(np.square(anova_feat_out_arr),axis=1))[:,np.newaxis]
                anova_feat_in_arr = anova_feat_in_arr/np.sqrt(np.sum(np.square(anova_feat_in_arr),axis=1))[:,np.newaxis]
                anova_feat_out_arr_pca = anova_feat_out_arr
                anova_feat_in_arr_pca = anova_feat_in_arr                
                ##################################### pca #################################################
                # calculate intra distance
                intra_distance = []
                for label in set(index_out_arr):
                    X0 = anova_feat_out_arr_pca_inver[np.where(index_out_arr==label)] 
                    sim0_intra = pairwise.pairwise_distances(X0, metric='cosine')
                    num = len(sim0_intra)
                    intra_distance.append(sim0_intra.sum()/(num*num-num))
                print(f'--Intra-- out_track_pca: max {max(intra_distance)}, min {min(intra_distance)}, average {sum(intra_distance)/len(intra_distance)}')
                
                intra_distance = []
                for label in set(index_in_arr):
                    X0 = anova_feat_in_arr_pca_inver[np.where(index_in_arr==label)] 
                    sim0_intra = pairwise.pairwise_distances(X0, metric='cosine')
                    num = len(sim0_intra)
                    intra_distance.append(sim0_intra.sum()/(num*num-num))
                print(f'--Intra-- in_track_pca: max {max(intra_distance)}, min {min(intra_distance)}, average {sum(intra_distance)/len(intra_distance)}')
                
                # calculate inter distance
                inter_distance = []
                clf = NearestCentroid()
                clf.fit(anova_feat_out_arr, index_out_arr)
                cen_dist = pairwise.pairwise_distances(clf.centroids_, metric='cosine')
                num = len(cen_dist)

                avrage_inter_disance = cen_dist.sum()/(num*num-num)
                max_inter_distance = np.max(cen_dist)
                for i in range(len(cen_dist)):
                    cen_dist[i,i] = 10e6
                min_inter_distance = np.min(cen_dist)
                print(f'--Inter-- out_track_pca: max {max_inter_distance}, min {min_inter_distance}, average {avrage_inter_disance}')
                self.nearest_pair(cen_dist)
                inter_distance = []
                clf = NearestCentroid()
                clf.fit(anova_feat_in_arr, index_in_arr)
                cen_dist = pairwise.pairwise_distances(clf.centroids_, metric='cosine')
                num = len(cen_dist)
                avrage_inter_disance = cen_dist.sum()/(num*num-num)
                
                max_inter_distance = np.max(cen_dist)
                for i in range(len(cen_dist)):
                    cen_dist[i,i] = 10e6
                min_inter_distance = np.min(cen_dist)
                print(f'--Inter-- in_track_pca: max {max_inter_distance}, min {min_inter_distance}, average {avrage_inter_disance}')

                
                ##################################### pca #################################################
                
                anova_feat_out_arr_ica = anova_feat_out_arr_ica/np.sqrt(np.sum(np.square(anova_feat_out_arr_ica),axis=1))[:,np.newaxis]
                anova_feat_in_arr_ica = anova_feat_in_arr_ica/np.sqrt(np.sum(np.square(anova_feat_in_arr_ica),axis=1))[:,np.newaxis]
                
                
                ##################################### ica #################################################
                intra_distance = []
                for label in set(index_out_arr):
                    X0 = anova_feat_out_arr_ica_inver[np.where(index_out_arr==label)] 
                    sim0_intra = pairwise.pairwise_distances(X0, metric='cosine')
                    num = len(sim0_intra)
                    intra_distance.append(sim0_intra.sum()/(num*num-num))
                    print(f'ica intra: {self.index_map_out[label]} {sim0_intra.sum()/(num*num-num)}')
                print(f'--Intra-- out_track_ica: max {max(intra_distance)}, min {min(intra_distance)}, average {sum(intra_distance)/len(intra_distance)}')
                intra_distance = []
                for label in set(index_in_arr):
                    X0 = anova_feat_in_arr_ica_inver[np.where(index_in_arr==label)] 
                    sim0_intra = pairwise.pairwise_distances(X0, metric='cosine')
                    num = len(sim0_intra)
                    intra_distance.append(sim0_intra.sum()/(num*num-num))
                print(f'--Intra-- in_track_ica: max {max(intra_distance)}, min {min(intra_distance)}, average {sum(intra_distance)/len(intra_distance)}')
                
                inter_distance = []
                clf = NearestCentroid()
                clf.fit(anova_feat_out_arr_ica_inver, index_out_arr)
                cen_dist = pairwise.pairwise_distances(clf.centroids_, metric='cosine')
                num = len(cen_dist)
               
                avrage_inter_disance = cen_dist.sum()/(num*num-num)
                max_inter_distance = np.max(cen_dist)
                for i in range(len(cen_dist)):
                    cen_dist[i,i] = 10e6
                min_inter_distance = np.min(cen_dist)
                print(f'--Inter-- out_track_ica: max {max_inter_distance}, min {min_inter_distance}, average {avrage_inter_disance}')
                self.nearest_pair(cen_dist)
                
                inter_distance = []
                clf = NearestCentroid()
                clf.fit(anova_feat_in_arr_ica_inver, index_in_arr)
                cen_dist = pairwise.pairwise_distances(clf.centroids_, metric='cosine')
                num = len(cen_dist)
                avrage_inter_disance = cen_dist.sum()/(num*num-num)
                max_inter_distance = np.max(cen_dist)
                for i in range(len(cen_dist)):
                    cen_dist[i,i] = 10e6
                min_inter_distance = np.min(cen_dist)
                print(f'--Inter-- in_track_ica: max {max_inter_distance}, min {min_inter_distance}, average {avrage_inter_disance}')

                anova_feat_out_arr = anova_feat_out_arr_ica
                anova_feat_in_arr = anova_feat_in_arr_ica              

        else:
            anova_feat_out_arr = feat_out_arr - np.mean(feat_out_arr,axis=0)
            anova_feat_in_arr = feat_in_arr - np.mean(feat_in_arr,axis=0)

            if len(list(set(index_out_arr)))>=5 and len(list(set(index_in_arr)))>=5: # data is sufficeient
                pca = PCA(n_components=0.9,random_state=2)    
                pca.fit(anova_feat_out_arr, index_out_arr) 
                anova_feat_out_arr = pca.transform(anova_feat_out_arr)
                anova_feat_in_arr = pca.transform(anova_feat_in_arr)    
                feat_out_arr = pca.inverse_transform(anova_feat_out_arr)
                feat_in_arr = pca.inverse_transform(anova_feat_in_arr)     
                anova_feat_out_arr = feat_out_arr - np.mean(feat_out_arr,axis=0)
                anova_feat_in_arr = feat_in_arr - np.mean(feat_in_arr,axis=0)
            else:
                print("skip pca")
                
            n = len(list(set(index_out_arr))) + common_feat
            ica = FastICA(n_components=n,whiten='arbitrary-variance',max_iter=10,random_state=2)
            ica.fit(anova_feat_out_arr, index_out_arr) 
            anova_feat_out_arr = ica.transform(anova_feat_out_arr)
            anova_feat_in_arr = ica.transform(anova_feat_in_arr)

            print("ica dim:",anova_feat_out_arr.shape[1])

        anova_feat_out_arr = anova_feat_out_arr/np.sqrt(np.sum(np.square(anova_feat_out_arr),axis=1))[:,np.newaxis]
        anova_feat_in_arr = anova_feat_in_arr/np.sqrt(np.sum(np.square(anova_feat_in_arr),axis=1))[:,np.newaxis]
        
        if name=="2":
            tmp = self.occ_out
            self.occ_out = self.occ_in
            self.occ_in = tmp
            self.blur_in = tmp
            tmp = self.index_map_out
            self.index_map_out = self.index_map_in
            self.index_map_in = tmp          

        
        return anova_feat_out_arr, anova_feat_in_arr
      
    def calculate_dist(self, anova_feat_out_arr, anova_feat_in_arr):
        q_q_sim = np.matmul(anova_feat_out_arr, anova_feat_out_arr.T)
        g_g_sim = np.matmul(anova_feat_in_arr, anova_feat_in_arr.T)
        q_g_sim = np.matmul(anova_feat_out_arr, anova_feat_in_arr.T)
        k1 = self.k1
        k2 = self.k2
        lambda_value = self.lambda_value
        dist_mat = re_ranking(q_g_sim, q_q_sim, g_g_sim, k1=k1, k2=k2, lambda_value=lambda_value) # nxm
        
        
        return dist_mat
        
    def zero(self):
        return 0
        
        
    def calculate_score(self, dist_mat_q_g, dist_mat_g_q, index_out_arr, index_in_arr, cam_out_id, cam_in_id, out_time_out_arr, in_time_in_arr, Iter=0):
        sorted_out_index_dist_mat = dist_mat_q_g.argsort(1) # nxm
  
        sorted_in_index_dist_mat = dist_mat_g_q.argsort(0) # nxm
        
        counter_dict_out = {}
        out_indexs, counts = np.unique(index_out_arr,return_counts=True)
        for ind, out_index in enumerate(out_indexs):
            counter_dict_out[out_index] = counts[ind]
        counter_dict_in = {}
        in_indexs, counts = np.unique(index_in_arr,return_counts=True)
        for ind, in_index in enumerate(in_indexs):
            counter_dict_in[in_index] = counts[ind]
            
        #counter_dict_out = defaultdict(self.zero)
        #counter_dict_in = defaultdict(self.zero)
        #for out_index in index_out_arr:
        #    counter_dict_out[out_index] += 1
        #for in_index in index_in_arr:
        #    counter_dict_in[in_index] += 1
        print(counter_dict_out)
        print(counter_dict_in)
        
        n_out = len(counter_dict_out)
        n_in = len(counter_dict_in)
        score_table = np.zeros((n_out,n_in))

        out_time_out_box_mat = np.expand_dims(out_time_out_arr, 1).repeat(n_in, 1) # n_out x n_in   
        in_time_in_box_mat = np.expand_dims(in_time_in_arr, 0).repeat(n_out, 0) # n_out x n_in     
        time_mat = in_time_in_box_mat-out_time_out_box_mat
        
            
        sim_mat_q_g = 1.0 - (dist_mat_q_g/np.max(dist_mat_q_g))
        sim_mat_g_q = 1.0 - (dist_mat_g_q/np.max(dist_mat_g_q))

        exp_sim_mat_q_g = np.exp(sim_mat_q_g)

        r_rate = self.r_rate[Iter]   
        topk = int(n_in*self.topk[Iter])
        topk2 = int(n_out*self.topk[Iter])
        if topk < 5:
            topk = 5
        if topk2 < 5:
            topk2 = 5
        
        if Iter == 0:
            print('use ALL')
        else:
            print('use length')
        time_match = time.time()
        for index_out_order in counter_dict_out.keys():
            focus_out = np.where(index_out_arr==index_out_order)[0]
            
            for out_index in focus_out: # out_index 0 run to 4X if track has 4X+1 imgs
                topk_columns = sorted_out_index_dist_mat[out_index,:topk] # get row[out_index] top-k column 'index' #values range from 0 to 6680
                 
                
                ####################
                weight_1 = exp_sim_mat_q_g[out_index,topk_columns]
                all_rows_topk_columns = sorted_in_index_dist_mat[:,topk_columns]
                
                reverse_sim = np.sort(sim_mat_g_q[:,topk_columns], axis=0) 
                reverse_sim = np.flipud(reverse_sim)
                reverse_sim = reverse_sim[:topk2,:]
                topk_rows_topk_columns = all_rows_topk_columns[:topk2,:]
                
                
                reverse_sim[np.isin(topk_rows_topk_columns,focus_out,invert=True)] = 0
                reverse_sim[np.where(topk_rows_topk_columns==out_index)] = 0
                
                reverse_sim = np.sort(reverse_sim, axis=0)
                count_nz = np.count_nonzero(reverse_sim, axis=0)

                if Iter==0:
                    reverse_sim[:,np.where(count_nz<3)[0]] = 0
                
                weight_2 = np.sum(np.square(reverse_sim),axis=0)[np.newaxis]/(np.sum(reverse_sim,axis=0)[np.newaxis]+1e-9) # 1.
                score = np.multiply(weight_1,weight_2)
                
                
                index_in_orders = np.unique([index_in_arr[val] for val in topk_columns.ravel()])
                
                for index_in_order in index_in_orders:
                    focus_in = np.where(index_in_arr==index_in_order)[0]
                    selected_columns = np.isin(topk_columns,focus_in).tolist()
                    length = len(topk_columns[selected_columns].tolist())
                    if Iter == 0:
                        if counter_dict_in[index_in_order]>0:
                            score_table[index_out_order,index_in_order] += np.sum(score[0,selected_columns])/(counter_dict_in[index_in_order]*counter_dict_out[index_out_order])
                    else:
                        if length > 1:
                            score_table[index_out_order,index_in_order] += np.sum(score[0,selected_columns])/(length*counter_dict_out[index_out_order])                
                

        score_table[time_mat<=0] = 0
        score_table[score_table<=r_rate] = 0
        print(time.time()-time_match)
        print("score calculation done")
        if Iter>0:
            print(score_table)

        matched_i = []
        matched_j = []

        time_assign = time.time()
        row_ind, col_ind = linear_sum_assignment(np.max(score_table)-score_table) 
        
        print(time.time()-time_assign)
        print("Assignment done")
        for i in range(len(row_ind)):
            print(row_ind[i],col_ind[i],score_table[row_ind[i],col_ind[i]])
            if score_table[row_ind[i],col_ind[i]] > r_rate:
                matched_i.append(row_ind[i])
                matched_j.append(col_ind[i])
  
        return np.array(matched_i), np.array(matched_j)
            

    def drop_invalid_matched_pairs(self, matched_i, matched_j, cam_out_id, cam_in_id, out_time_out_arr, in_time_in_arr):
        """
        Args:
            matched_i: 
            matched_j:
        """
        tth_min, tth_max = two_track_valid_pass_time_dict[(cam_out_id, cam_in_id)]
        keep_ids = []
        for idx, (i, j) in enumerate(zip(matched_i, matched_j)):
            travel_time = in_time_in_arr[j] - out_time_out_arr[i]
            if travel_time < tth_min or travel_time > tth_max:
                continue
            keep_ids.append(idx)
        matched_i = matched_i[keep_ids]
        matched_j = matched_j[keep_ids]
        return matched_i, matched_j

    def matching(self, cam_in_id, cam_out_id, interval_out=[0, 2001], interval_in=[0, 2001], \
                direction=True, mode='linear', st_dim=0, en_dim=2048, is_params=True):
        if cam_in_id > cam_out_id:
            assert direction == True
        else:
            assert direction == False

        if is_params:
            map_out_arr = self.select_map_arr_interval(cam_out_id, is_out=True, \
                            interval=interval_out, direction=direction)
        else:
            map_out_arr = self.select_map_arr(cam_out_id, is_out=True, direction=direction)
        cam_out_arr = self.cam_arr[map_out_arr]
        track_out_arr = self.track_arr[map_out_arr]
        in_time_out_arr = self.in_time_arr[map_out_arr]
        out_time_out_arr = self.out_time_arr[map_out_arr]
        

        if is_params:
            map_in_arr = self.select_map_arr_interval(cam_in_id, is_out=False, \
                            interval=interval_in, direction=direction)
        else:
            map_in_arr = self.select_map_arr(cam_in_id, is_out=False, direction=direction)
        cam_in_arr = self.cam_arr[map_in_arr]
        track_in_arr = self.track_arr[map_in_arr]
        in_time_in_arr = self.in_time_arr[map_in_arr]
        out_time_in_arr = self.out_time_arr[map_in_arr]
        
        
        print ('cam: {}; tracks: {} \t track_ids: {}'.format(cam_out_id, len(track_out_arr), np.sort(track_out_arr)))
        print ('cam: {}; tracks: {} \t track_ids: {}'.format(cam_in_id, len(track_in_arr), np.sort(track_in_arr)))


        #### Search results circularly
        all_matched_i = []
        all_matched_j = []
        print ('* Start matching...')
        topk = self.topk
        r_rate = self.r_rate
        num_search_times = self.num_search_times
        print(num_search_times)
        for i in range(num_search_times):
            print ('** Iter {}...'.format(i))
            if self.r_rate[i]==10:
                break
            sub_track_out_arr = np.setdiff1d(track_out_arr, track_out_arr[all_matched_i]) # sorted. need to be readjust
            sub_track_in_arr = np.setdiff1d(track_in_arr, track_in_arr[all_matched_j]) # sorted. need to be readjust
            num_candidates = 3
            if sub_track_out_arr.shape[0] < num_candidates or sub_track_in_arr.shape[0] < num_candidates:
                break

            map_sub_out_arr = np.isin(track_out_arr, sub_track_out_arr, True) # accelerate
            map_sub_in_arr = np.isin(track_in_arr, sub_track_in_arr, True) # accelerate

            sub_track_out_arr = track_out_arr[map_sub_out_arr] # original order
            sub_track_in_arr = track_in_arr[map_sub_in_arr] # original order

            sub_cam_out_arr = cam_out_arr[map_sub_out_arr]
            sub_cam_in_arr = cam_in_arr[map_sub_in_arr]

            sub_out_time_out_arr = out_time_out_arr[map_sub_out_arr]
            sub_in_time_in_arr = in_time_in_arr[map_sub_in_arr]

            r = min(sub_track_out_arr.shape[0] / float(track_out_arr.shape[0]), 
                    sub_track_in_arr.shape[0] / float(track_in_arr.shape[0]))
            print(sub_track_out_arr) 
            print(sub_track_in_arr)
            self.index_map_out = sub_track_out_arr
            self.index_map_in = sub_track_in_arr
            sub_matched_i, sub_matched_j = self.do_matching(sub_cam_out_arr, sub_cam_in_arr, sub_track_out_arr, 
                                                    sub_track_in_arr, sub_out_time_out_arr, sub_in_time_in_arr, 
                                                    cam_out_id, cam_in_id, st_dim=st_dim, en_dim=en_dim, Iter=i)
            print("matching time: ",time.time()-self.start_time)
            sub_matched_i, sub_matched_j = self.drop_invalid_matched_pairs(sub_matched_i, sub_matched_j, 
                                                cam_out_id, cam_in_id, sub_out_time_out_arr, sub_in_time_in_arr)
            
            for smi, smj in zip(sub_matched_i, sub_matched_j):
                # assert mi, mj only match one item in track_arr
                mi = np.where(track_out_arr == sub_track_out_arr[smi])[0].item()
                mj = np.where(track_in_arr == sub_track_in_arr[smj])[0].item()
                
                assert (mi not in all_matched_i)
                assert (mj not in all_matched_j)
                all_matched_i.append(mi)
                all_matched_j.append(mj)
        print(all_matched_i)
        matched_i = np.array(all_matched_i)
        matched_j = np.array(all_matched_j)

        matched_track_out_arr = track_out_arr[matched_i]
        sorted_ids = np.argsort(matched_track_out_arr)
        matched_i = matched_i[sorted_ids] # for print
        matched_j = matched_j[sorted_ids] # for print

        print ('number of matched pairs: {}'.format(len(matched_i)))
        global_max_id = self.global_id_arr.max() + 1


        for i, j in zip(matched_i, matched_j):
            track_out_id = track_out_arr[i]
            track_in_id = track_in_arr[j]

            print("track_out_id",track_out_id,i)
            print("track_in_id",track_in_id,j) 
            idx_i = (self.cam_arr == cam_out_id) & (self.track_arr == track_out_id)
            idx_j = (self.cam_arr == cam_in_id) & (self.track_arr == track_in_id)

            try:
                assert (self.global_id_arr[idx_j].item() == -1)
            except:
                pdb.set_trace()
            if self.global_id_arr[idx_i].item() != -1:
                self.global_id_arr[idx_j] = self.global_id_arr[idx_i]
            else:
                self.global_id_arr[idx_i] = global_max_id
                self.global_id_arr[idx_j] = global_max_id
                global_max_id += 1
            all_g_ids = np.where(self.global_id_arr == self.global_id_arr[idx_i].item())[0]
            all_matched_cams = self.cam_arr[all_g_ids]
            all_matched_tracks = self.track_arr[all_g_ids]
            print ('{:3d}: ({:3d}, {:3d}) \t interval: {:4d} \t all_matched_cams: {:18s} \t '
                    'all_matched_tracks: {}'.format(self.global_id_arr[idx_i].item(), 
                    track_out_id, track_in_id, self.in_time_arr[idx_j].item() - self.out_time_arr[idx_i].item(), 
                    ', '.join(map(str, all_matched_cams)), ', '.join(map(str, all_matched_tracks))))

        
    def forward_matching(self, mode='linear', st_dim=0, en_dim=2048):
        # positve matching
        for cam_id in range(41, 46):
            #continue
            #cam_id = 41
            cam_out_id = cam_id
            cam_in_id = cam_id + 1
            print ('out: {}; in: {}'.format(cam_out_id, cam_in_id))

            key = (cam_out_id, cam_in_id)
            print ('params: {}'.format(args_params_dict[key]))
            self.topk = args_params_dict[key]['topk']
            self.r_rate = args_params_dict[key]['r_rate']
            self.k1 = args_params_dict[key]['k1']
            self.k2 = args_params_dict[key]['k2']
            self.lambda_value = args_params_dict[key]['lambda_value']
            self.alpha = args_params_dict[key]['alpha']
            self.long_time_t = args_params_dict[key]['long_time_t']
            self.short_time_t = args_params_dict[key]['short_time_t']
            self.use_ica = args_params_dict[key]['use_ica']
            self.num_search_times = args_params_dict[key]['num_search_times']
            
            self.matching(cam_in_id, cam_out_id, direction=True, mode=mode, \
                    st_dim=st_dim, en_dim=en_dim, is_params=False)
            #exit()        
        # negative matching
        for cam_id in range(46, 41, -1):
            cam_out_id = cam_id
            cam_in_id = cam_id - 1
            print ('out: {}; in: {}'.format(cam_out_id, cam_in_id))

            key = (cam_out_id, cam_in_id)
            print ('params: {}'.format(args_params_dict[key]))
            self.topk = args_params_dict[key]['topk']
            self.r_rate = args_params_dict[key]['r_rate']
            self.k1 = args_params_dict[key]['k1']
            self.k2 = args_params_dict[key]['k2']
            self.lambda_value = args_params_dict[key]['lambda_value']
            self.alpha = args_params_dict[key]['alpha']
            self.long_time_t = args_params_dict[key]['long_time_t']
            self.short_time_t = args_params_dict[key]['short_time_t']
            self.use_ica = args_params_dict[key]['use_ica']
            self.num_search_times = args_params_dict[key]['num_search_times']
            
            self.matching(cam_in_id, cam_out_id, direction=False, mode=mode, \
                    st_dim=st_dim, en_dim=en_dim, is_params=False)
            #exit()

    def write_output(self, src_path, dst_path):
        if not os.path.exists(os.path.dirname(dst_path)):
            os.makedirs(os.path.dirname(dst_path))

        print ('* writing output...')
        dst_obj = open(dst_path, 'w')
        with open(src_path, 'r') as fid:
            for line in fid.readlines():
                s = [int(i) for i in line.rstrip().split()]

                if s[0] == 45 or s[0] == 46:
                    h = 720
                else:
                    h = 960
                w = 1280
                if s[3] < 0 or s[4] < 0 or (s[5]+s[3]) > w or (s[6]+s[4]) > h:
                    continue

                idx = ((self.cam_arr == s[0]) & (self.track_arr == s[1]))
                g_id = self.global_id_arr[idx].item()
                if g_id != -1:
                    s[1] = g_id
                    
                    dst_obj.write('{}\n'.format(' '.join(map(str, s)))) # [camera_id, track_id, frame_id, x, y, w, h, -1, -1]
        dst_obj.close()


def prepare_data(cam_path, track_path, in_out_all_path, feat_path=None):
    cam_arr = np.load(cam_path)
    track_arr = np.load(track_path)
    feat_arr = np.load(feat_path) if feat_path is not None else np.zeros((cam_arr.shape[0], 8))
    c_dict = {}
    with open(in_out_all_path, 'r') as fid:
        for line in fid.readlines():
            s = [int(i) for i in line.rstrip().split()]
            if s[0] not in c_dict:
                c_dict[s[0]] = {}
            c_dict[s[0]][s[1]] = s[2:]

    sorted_ids = np.argsort(cam_arr)
    cam_arr = cam_arr[sorted_ids]
    track_arr = track_arr[sorted_ids]
    feat_arr = feat_arr[sorted_ids]
    in_dir_arr = np.zeros_like(cam_arr)
    out_dir_arr = np.zeros_like(cam_arr)
    in_time_arr = np.zeros_like(cam_arr)
    out_time_arr = np.zeros_like(cam_arr)
    for i in range(len(cam_arr)):
        in_dir_arr[i], out_dir_arr[i], in_time_arr[i], out_time_arr[i] = c_dict[cam_arr[i]][track_arr[i]]
    return cam_arr, track_arr, in_dir_arr, out_dir_arr, in_time_arr, out_time_arr, feat_arr,

def load_feat_dict(feat_path):
    feat_dict = pickle.load(open(feat_path, 'rb'))
    return feat_dict

def load_pickle_dict(pickle_path):
    pickle_dict = pickle.load(open(pickle_path, 'rb'))
    return pickle_dict

if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    s_time = time.time()
    cam_path = os.path.join(args.src_root, 'cam_vec.npy')
    track_path = os.path.join(args.src_root, 'track_vec.npy')
    in_out_all_path = os.path.join(args.src_root, 'in_out_all.txt')
    feat_path = None # the original version, it has been deprecated.
    feat_path2 = os.path.join(args.src_root, 'feat_all_vec.pkl')
    
    fr_path = os.path.join(args.src_root, 'frame_all_vec.pkl')
    fr_dict = load_pickle_dict(fr_path)
    src_path = os.path.join(args.src_root, 'all_cameras.txt')
    dst_path = os.path.join(args.dst_root, 'track1.txt')
    if not os.path.exists(args.dst_root):
        os.makedirs(args.dst_root)

    feat_dict = load_feat_dict(feat_path2) # load features

    trun_path = os.path.join(args.src_root, 'truncation.pkl')
    trun_dict = load_pickle_dict(trun_path) # load truncation rates

    track_dict = {}
    with open(src_path, 'r') as f_scmt:
        for line in f_scmt.readlines():
            s = [int(i) for i in line.rstrip().split()]
            if s[0] not in track_dict:
                track_dict[s[0]] = {}
            if s[2] not in track_dict[s[0]]:
                track_dict[s[0]][s[2]] = {}
            track_dict[s[0]][s[2]][s[1]] = s[3:]

    preprocessed_data = prepare_data(cam_path, track_path, in_out_all_path) # load all preprocessed data
    cam_arr, track_arr, in_dir_arr, out_dir_arr, in_time_arr, out_time_arr, feat_arr = preprocessed_data

    matcher = MultiCameraMatching(cam_arr, track_arr, in_dir_arr, out_dir_arr, 
                                in_time_arr, out_time_arr, feat_dict, track_dict,
                                topk=args.topk, r_rate=args.r_rate, 
                                k1=args.k1, k2=args.k2, lambda_value=args.lambda_value,
                                alpha=args.alpha, long_time_t=args.long_time_t, short_time_t=args.short_time_t,
                                num_search_times=args.num_search_times, fr_dict = fr_dict,
                                trun_dict=trun_dict, occ_rate=args.occ_rate, occ_alpha=args.occ_alpha)

    matcher.forward_matching(mode=args.mode, st_dim=args.st_dim, en_dim=args.en_dim)

    print ('* matching done.')
    print(time.time()-s_time)
    matcher.write_output(src_path, dst_path)
