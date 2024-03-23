import os
import pdb
import numpy as np
import pickle
import paddle
import paddle.nn.functional as F

query = pickle.load(open('dets_feat.pkl', 'rb'), encoding='latin1')
#query = pickle.load(open('aic22_1_test_infer_v2_ResNext101.pkl', 'rb'), encoding='latin1')
fuse_fea = {}
print(len(list(query.keys())))

def adjust(query):
    for each in query:
        feat_norm = np.sqrt(np.sum(np.square(query[each])))
        query[each] = np.divide(query[each],feat_norm)
    return query


def check_one_feature(feat):
    print(np.mean(feat))    
    print(np.var(feat))
    
def check_one_len(feat):
    print(np.sqrt(np.sum(np.square(feat))))
    
check = ['1','2','3','4']
check = ['3','4']
    
all_feat = []

query = adjust(query)
with open('./aic22_uda.pkl','wb') as fid:
    pickle.dump(query, fid)
for each in query:
    if '1' in check:
        check_one_feature(query[each])
        print(f'pass 1: {each}')
    
    if '2' in check:
        check_one_len(query[each])
        print(f'pass 2: {each}')
    all_feat.append(query[each])
arr_feat = np.array(all_feat)
 

m,n = np.shape(arr_feat)
for i in range(n):      # channel wise check
    if '3' in check:
        check_one_feature(arr_feat[:,i])
    if '4' in check:
        check_one_len(arr_feat[:,i])    
    exit() 
    
print('done')
    



