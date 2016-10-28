#-*- coding:utf-8 -*-

'''
Created on 2016年10月27日

@author: wangzhe
'''

import numpy as np

def load_data(fileName):      
    datas = []               
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine)
        datas.append(fltLine)
    return np.array(datas)

class Cart_rt():
     
    @staticmethod
    def reg_err(datas):
        return np.var(datas[:,-1])*datas.shape[0]
    
    def reg_leaf(self,datas):
        return np.mean(datas[:,-1])

    def bin_split_data(self,datas,feat_idx,feat_val):
         
        #nonzero 返回满足条件的矩阵的行，列
        idx = datas[:,feat_idx] <= feat_val
        data_left = datas[idx]
        data_right = datas[~idx]
        
        return data_left,data_right
    
    def choose_best_split(self,datas,ops=(1,4)):
        
        tol_s = ops[0]
        tol_n = ops[1]
        
        if len(set(datas[:,-1])) == 1:
            return None,self.reg_leaf(datas)
        
        cols = datas.shape[1]
        
        S = self.reg_err(datas)
        best_s = np.inf
        best_idx = 0
        best_val = 0.0
        
        for col in range(cols-1):
            for feat in set(datas[:,col]):
                datas_left,datas_right = self.bin_split_data(datas, col, feat)
                if datas_left.shape[0] < tol_n or datas_right.shape[0] < tol_n:
                    continue
                new_s  = self.reg_err(datas_left) + self.reg_err(datas_right)
                if new_s < best_s:
                    best_idx = col
                    best_val = feat
                    best_s = new_s
        if S - best_s < tol_s:
            return None,self.reg_leaf(datas)
        
        datas_left,datas_right = self.bin_split_data(datas, best_idx, best_val)
        if datas_left.shape[0] < tol_n or datas_right.shape[0] < tol_n:
            return None,self.reg_leaf(datas)
         
        return best_idx,best_val
    
    def create_tree(self,datas,ops=(1,4)):
        ss,val = self.choose_best_split(datas, ops)
        if ss == None:
            return val
        ret_tree = {}
        ret_tree['split'] = ss
        ret_tree['ssVal'] = val
        ldatas,rdatas = self.bin_split_data(datas, ss, val)
        ret_tree['left'] = self.create_tree(ldatas, ops)
        ret_tree['right'] = self.create_tree(rdatas, ops)
        return ret_tree
    
    def is_tree(self,obj):
        return (type(obj).__name__ == 'dict')
    
    def get_mean(self,tree):
        if self.is_tree(tree['right']):
            tree['right'] = self.get_mean(tree['right'])
        if self.is_tree(tree['left']):
            tree['left'] = self.get_mean(tree['left'])
        return (tree['left']+tree['right'])/2.0
    
    def prune(self,tree,tests):
        
        if tests.shape[0] == 0:
            return self.get_mean(tree)
        if self.is_tree(tree['right']) or self.is_tree(tree['left']):
            ldatas,rdatas = self.bin_split_data(tests, tree['split'], tree['ssVal'])
        if self.is_tree(tree['left']):
            tree['left'] = self.prune(tree['left'], ldatas)
        if self.is_tree(tree['right']):
            tree['right'] = self.prune(tree['right'],rdatas)
        if not self.is_tree(tree['left']) and not self.is_tree(tree['right']):
            ldatas,rdatas = self.bin_split_data(tests, tree['split'],tree['ssVal'])
            error_no_merge = sum(np.power(ldatas[:,-1]-tree['left'],2)) +\
                             sum(np.power(rdatas[:,-1]-tree['right'],2))
            tree_mean = (tree['left']+tree['right'])/2.0
            
            error_merge = sum(np.power(tests[:,-1]-tree_mean,2)) 
            if error_merge < error_no_merge:
                print "merging"
                return tree_mean
            else:
                return tree
        else:
            return tree

if __name__ == "__main__":
    
    datas = load_data("e:/data/ex0.txt")
    ccr = Cart_rt()
    tree = ccr.create_tree(datas)
    tests = load_data("e:/data/ex2test.txt")
    ccr.prune(tree, tests)
    
    
    
    
