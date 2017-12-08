#! -*- coding:utf-8 -*-


'''
此代码的思想是创建一个堆，根据查找二叉树和回溯的节点到test节点的大小及堆内的节点的个数，
   来维护堆的大小始终为k，并且这k个节点都是距离test节点最近的k个点。
'''

import numpy as np
from collections import Counter
from sklearn.cross_validation import train_test_split


class Node:
    
    def __init__(self, point=None, label=None, split=None, parent=None,
                 left=None, right=None, dist=None):
        
        self.point = point
        self.label = label
        self.parent = parent
        self.left = left
        self.right = right
        self.split = split
        self.dist = dist


class KD_tree:
    
    @staticmethod 
    def get_split(X):
    
        mean_v =np.mean(X, axis=0)
        val = sum((X-mean_v)**2)/float(X.shape[0])
        idxs = val.argsort()
        return idxs[-1]
    
    def create_kdtree(self,datas,labels):

        if datas.shape[0] == 0:
            return 
        
        N = datas.shape[0]
        #计算各个维度的方差，以最大方差的维度作为切分维度
        split = self.get_split(datas)
        #在切分维度上以中位数的点为切分点
        split_sort = datas[:,split].argsort()
        point = datas[split_sort[N/2],:]
        label = labels[split_sort[N/2]]
        root = Node(point,label,split)
        root.left = self.create_kdtree(datas[split_sort[:N/2]],
                                       labels[split_sort[:N/2]])
        root.right = self.create_kdtree(datas[split_sort[N/2+1:]],
                                        labels[split_sort[N/2+1:]])
        
        if root.left != None:
            root.left.parent = root
        if root.right != None:
            root.right.parent = root
        return root
    
    @staticmethod
    def compute_dist(X1,X2):
    
        return np.sqrt(sum((X1-X2)**2))
    
    def search_leaf(self,query,root):
    
        leaf = root
        sub_node = None
        
        while(leaf.left != None or leaf.right != None):
            ss = leaf.split
            if query[ss] < leaf.point[ss]:
                sub_node = leaf.left
            elif query[ss] > leaf.point[ss]:
                sub_node = leaf.right
            else:
                if self.compute_dist(query,leaf.point) <= self.compute_dist(query,
                                          leaf.right):
                    sub_node = leaf.left
                else:
                    sub_node = leaf.right
            
            if sub_node == None:
                break
            else:
                leaf = sub_node
        return leaf        
    
    @staticmethod
    def get_brother(node):
        if id(node) == id(node.parent.left):
            return node.parent.right
        else:
            return node.parent.left
    
    def search_knn(self,query,root,k):
    
        knn_list = []    
        almost_node = self.search_leaf(query, root) #在kd树种找到test数据的叶子节点
        heap = Heap(k,knn_list) # 初始化堆
        
        while( almost_node != None):
            
            cur_dist = self.compute_dist(query,almost_node.point)  #计算test数据与叶子节点的距离
            almost_node.dist = cur_dist
            heap.adjust_heap(almost_node)

            if almost_node.parent != None and abs(query[almost_node.parent.split] -
                                                  almost_node.parent.point[almost_node.parent.split]) < cur_dist:
                
                brother = self.get_brother(almost_node)
                if brother != None:
                    brother.dist = self.compute_dist(query, brother.point)
                    heap.adjust_heap(brother)
            almost_node = almost_node.parent
         
        return knn_list
            
            
class Heap:
    
    def __init__(self,k=0,list_node=[]):
        
        self.k = k
        self.list_node = list_node
    
    def adjust_heap(self,node):
        if len(self.list_node) < self.k:
            self.max_heap_fixup(node) # 如果堆中的点个数小于k，那么加入新的节点，调整堆结构
        elif node.dist < self.list_node[0].dist:
            self.max_heap_fixdown(node) #如果堆中点个数大于k，比较目前的最近距离与堆中最短距离的距离大小，决定是否把这个点加入堆中。
    
    def max_heap_fixup(self,new_node):
        
        self.list_node.append(new_node)
        
        j = len(self.list_node) - 1
        i = (j+1)/2 - 1
        
        while i >= 0:
            if self.list_node[i].dist >= self.list_node[j].dist:
                break    
            self.list_node[i],self.list_node[j] = self.list_node[j],self.list_node[i]
             
            j=i
            i=(j+1)/2-1
            
    def max_heap_fixdown(self,new_node):
        
        self.list_node[0] = new_node
        
        i = 0
        j = i*2+1
        
        while(j < len(self.list_node)):
            if j+1 < len(self.list_node) and self.list_node[j].dist < self.list_node[j+1].dist:
                j += 1
            
            if self.list_node[i].dist >= self.list_node[j].dist:
                break
            self.list_node[i],self.list_node[j] = self.list_node[j],self.list_node[i]
             
            i=j
            j=i*2+1
            
class Knn:
     
    def __init__(self,k,datas,labels):
        
        self.k = k
        self.datas = datas
        self.labels = labels
    
    def predict(self,tests):
        
        if len(self.datas) == 0 :
            raise IOError,"imput train data error"
        if len(tests) == 0:
            raise IOError,"input test data error"
        if len(self.datas) != len(self.labels):
            raise IOError,"datas size is not equals labels size"
        
        preds = []
        kdtree = KD_tree()
        root = kdtree.create_kdtree(self.datas, self.labels)  # 创建一个kd树
        for test in tests:
            knn_list = kdtree.search_knn(test,root,self.k)  # 查找距离最近的k个节点
            pred_labels = map(lambda x:x.label,knn_list)    #节点的标签
            pred_count = Counter(pred_labels)                #统计k个节点的标签
            pred_count = sorted(pred_count.iteritems(),key=lambda x:x[1],reverse=True)  # 按照标签的统计值从大到小排序
            pred = pred_count[0][0]
            preds.append(pred)
        return preds
    
    def estimate(self,test_labels,pred_labels):
        
        if len(test_labels) != len(pred_labels):
            raise IOError,"Data inconsistency"
        
        accuracy = 0.0 
        for idx,val in enumerate(test_labels):
            if val == pred_labels[idx]:
                accuracy += 1
        accuracy = accuracy/len(test_labels)
        print "accuracy : %f"%(accuracy)
        
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
    norm_datas = datas - np.tile(min_v,(datas.shape[0],1))
    norm_datas = norm_datas/np.tile(max_v-min_v,(datas.shape[0],1))
    return norm_datas

if __name__ == "__main__":
    
    filename = "E:/data/datingTestSet2.txt"
    datas, labels = load_data(filename)
    norm_datas = auto_norm(datas)
    
    x_tr, x_tt, y_tr, y_tt = train_test_split(norm_datas,labels, train_size=.60, random_state=10)
    
    kdtree = KD_tree()
    knn = Knn(6,x_tr,y_tr)
    preds = knn.predict(x_tt)
    knn.estimate(y_tt, np.array(preds))
