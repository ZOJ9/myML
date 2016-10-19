#-*- coding:utf-8 -*-

import numpy as np
import operator

def load_data(filename):
    
    labels = []
    datas = []
    fr = open(filename)
    for line in fr :
        lines = line.strip().split("\t")
        lines = map(float,lines)
        datas.append(lines[:-1])
        labels.append(lines[-1])
    return np.array(datas),np.array(labels)

def auto_norm(datas):
    
    min_v = datas.min(0)
    max_v = datas.max(0)
#     norm_datas = np.zeros(datas.shape)
    norm_datas = datas - np.tile(min_v,(datas.shape[0],1))
    norm_datas = norm_datas/np.tile(max_v-min_v,(datas.shape[0],1))
    return norm_datas

def classify0(test, datas, labels, k):
    
    m = datas.shape[0]
    diff_mat = np.tile(test,(m,1))-datas
    sqrt_mat = diff_mat**2
    dist_mat = sqrt_mat.sum(axis=1)
    dist_mat = dist_mat**0.5
    dist_mat = dist_mat.argsort()
    
    class_count = {}
    for i in range(k):
        vote_label = labels[dist_mat[i]]
        class_count[vote_label] = class_count.get(vote_label,0)+1
    sort_class = sorted(class_count.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sort_class[0][0]

def train_test(datas, labels, k):
    
    ratio = 0.5
    m = datas.shape[0]
    error_count = 0.0
    test_nums = int(m*ratio)
    
    for i in range(test_nums):
        ret_label = classify0(datas[i,:], datas[test_nums:m], labels[test_nums:m], k)
        if ret_label != labels[i]:
            error_count += 1.0
    print "error rate : %f"%(error_count/float(test_nums))

if __name__ == "__main__":
    
    filename = "E:/data/datingTestSet2.txt"
    datas,labels = load_data(filename)
    norm_datas = auto_norm(datas)
    train_test(norm_datas, labels, 3)
    
