#-*- coding:utf-8 -*-

'''
Created on 2016年10月21日

@author: wangzhe
'''

import numpy as np
from collections import Counter

class DT_TREE:
    
    def __init__(self,score,alpha):
        self.score = score
        self.count = 0
        self.alpha = alpha
        
    class tree_node():
        def __init__(self,labels=None):
            self.split = np.inf
            self.parent = None
            self.childs = {}
            self.leaflable = labels
            self.leafdatas = None
            self.leafscore = 0.0
    
    @staticmethod
    def calcu_entropy(labels):
        c_count_dict = Counter(labels)
        c_count_lst = c_count_dict.values()
        c_prob = map(lambda x:x/float(len(labels)),c_count_lst)
        c_entropy = map(lambda x:x*np.log2(x),c_prob)
        entropy = -1.0*sum(c_entropy)
        return entropy
    
    @staticmethod
    def gini_coef(labels):
        c_count_dict = Counter(labels)
        c_count_lst = c_count_dict.values()
        c_prob = map(lambda x:np.square(x/float(len(labels))),c_count_lst)
        gini = 1.0-sum(c_prob)
        return gini        
    
    #计算信息增益
    def info_gain(self,feat_datas,labels,H_D):
        
        H_F_D = 0.0
        feats = set(feat_datas)

        for f in feats:
            f_idx = feat_datas==f
            f_datas = feat_datas[f_idx]
            f_labels = labels[f_idx]
            h_f = self.calcu_entropy(f_labels)
            H_F_D += (float(len(f_datas))/len(feat_datas))*h_f  
        gain = H_D - H_F_D
        return gain
    
    #计算信息增益比
    def info_gain_ratio(self,feat_datas,labels,H_D):
        
        H_F_D = 0.0
        feats = set(feat_datas)

        for f in feats:
            f_idx = feat_datas==f
            f_datas = feat_datas[f_idx]
            f_labels = labels[f_idx]
            h_f = self.calcu_entropy(f_labels)
            H_F_D += (float(len(f_datas))/len(feat_datas))*h_f  
        gain = H_D - H_F_D
        gain_ratio = gain/H_D
        return gain_ratio

    #计算基尼系数
    def condition_gini_coef(self,feat_datas,labels):
        
        G_F_D = 0.0
        feats = set(feat_datas)

        for f in feats:
            f_idx = feat_datas==f
            f_datas = feat_datas[f_idx]
            f_labels = labels[f_idx]
            g_f = self.gini_coef(f_labels)
            G_F_D += (float(len(f_datas))/len(feat_datas))*g_f  
        return G_F_D
    
    #选择最优特征
    def choose_best_feature(self,datas,labels):
        
        best_idx = 0
        max_score = 0.0
        
        if self.score == "gain" or self.score == "rgain":
            H_D = self.calcu_entropy(labels)
        
        for idx in range(datas.shape[1]):
            feat_datas = datas[:,idx]
            if self.score == "gain":
                score = self.info_gain(feat_datas, labels, H_D)
            elif self.score == "rgain":
                score = self.info_gain_ratio(feat_datas, labels, H_D)
            elif self.score == "gini":
                score = self.condition_gini_coef(feat_datas, labels)
            else:
                raise IOError,"please input metrics"
            if score >= max_score:
                max_score = score
                best_idx = idx
        return best_idx,max_score
    
    #创建树
    def create_tree(self,datas,labels):
        
        if len(datas) != len(labels):
            raise IOError,"the length of the inconsistent for datas and label"
        
        root = self.tree_node(labels)
        
        #减枝使用"H(T_t)*叶节点的样本点个数"
        root.leafscore = self.calcu_entropy(labels)*len(labels)
        
        best_split,max_scores = self.choose_best_feature(datas,labels)
        if max_scores <= 0:
            self.count += 1
            root.leafdatas = datas
            return root
        
        root.split = best_split
        feats = datas[:,best_split]
        datas = np.delete(datas,best_split,axis=1)

        for feat in set(feats):
            f_idx =  feats == feat
            root.childs[feat] = self.create_tree(datas[f_idx],labels[f_idx])
            root.childs[feat].parent = root
        return root
    
    #进行预测
    def pred(self,test,root):
        
        if len(root.childs) == 0:
            l_count = Counter(root.leaflable)
            l_sort = sorted(l_count.items(),key=lambda x:x[1],reverse=True)
            return l_sort[0][0]
        
        split = root.split
        feat = test[split]
         
        test = np.delete(test,split)
        if feat in root.childs:
            return self.pred(test,root.childs[feat])
        
    #统计树的叶子节点
    def get_leaf_count(self,tree):
        
        nums = 0
        childs = tree.childs
        if len(childs) != 0:
            for k in childs.values():
                nums += self.get_leaf_count(k)
        else:
            nums += 1
        return nums
    
    #统计树的深度
    def get_depth(self,tree):
        
        nums = 0
        childs = tree.childs
        if len(childs) != 0:
            for k in childs.values():
                thisnums = 1+ self.get_depth(k)
        else:
            thisnums = 1
        if thisnums > nums :
            nums = thisnums
        return nums
    
    #减枝
    def prune(self,tree):
        
        is_prune = False
         
        if len(tree.childs) == 0:
            
            parent_childs = tree.parent.childs
            parent_count = 0
            parent_labels = []
            
            for v in parent_childs.values():
                parent_count += self.get_leaf_count(v)
                parent_labels += v.leaflable
            
            current_count = self.get_leaf_count(tree)
            current_loss = current_count * self.alpha + tree.leafscore
             
            parent_loss = (current_count+1-parent_count) *\
                            self.alpha + tree.parent.leafscore
             
            if parent_loss <= current_loss:
                tree.parent.childs = {}
                tree.leaflable = parent_labels
                is_prune = True
        else:
            if is_prune:
                self.prune(tree.parent)
            else:
                for v in tree.childs.values():
                    self.prune(v)
            
                   
                 
if __name__ == "__main__":
    
    datas = [["youth","noWork","noHouse","bad"],
             ["youth","noWork","noHouse","good"],
             ["youth","work","noHouse","good"],
             ["youth","work","house","bad"],
             ["youth","noWork","noHouse","bad"],
             ["middle","noWork","noHouse","bad"],
             ["middle","noWork","noHouse","good"],
             ["middle","work","house","good"],
             ["middle","noWork","house","best"],
             ["middle","noWork","house","best"],
             ["old","noWork","house","best"],
             ["old","noWork","house","good"],
             ["old","work","noHouse","good"],
             ["old","work","noHouse","best"],
             ["old","noWork","noHouse","bad"]]
    
    labels = ["no","no","yes","yes","no","no","no",
              "yes","yes","yes","yes","yes","yes","yes","no"]
    
    datas = np.array(datas)
    labels = np.array(labels)
    dt = DT_TREE("gain",0.2)
    root = dt.create_tree(datas,labels)
    dt.prune(root)
    l = dt.pred(datas[2], root)
    print l
    
    
    
    

