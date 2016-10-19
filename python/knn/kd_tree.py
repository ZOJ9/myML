#-*- coding:utf-8 -*-

'''
KD树的思想很简单，就是分割空间：
  1. 首先建立一个高维的二叉树，因为是高维就需要考虑从哪个维度切割空间（选取根节点），
             一般的选择是选取方差比较大的维度作为切割维度，方差大说明再这个维度上的数据分布比较分散。
             选好了切割维度，那么需要选择切割点了，即就是在这个维度上的哪个点开始切割，一般选取这个维度上的数据的中位数。
             切割面是垂直于切割维度面的超平面。
  2. kd树建好后下面就需要搜索最近邻的点。
     1） 搜索的时候我们从树的根节点开始，在这个维度（建立树的时候需要记录每个节点的切割维度）
                    上的数据小于此节点在这个维度上的数，那么进入左子树，否则进入右子树。
                     此过程可以建立一个栈（后进先出）记录搜索过程中遍历的节点，同时记录在这些节点中的最小距离。
     2）回溯
                   为什么要有这步，简单的想一下，在第一步的时候，如果搜索数据在维度a上的数小于节点在此维度上的数，
                   那么搜索就会进入到左子树继续搜索，那么想想，数据在a维度上小并不代表它在整个维度上的最近邻点就会落入到左子树区域里
                   判断的标准是，在回溯到某一节点（这些节点就是遍历的点，也就是栈中记录的点）的时候，如果
                   搜索点与改节点在切割维度上的距离l（也就是点在次维度上到平面的距离了） 小于  最小距离d（这句话的意思就是以搜索点为圆心
                   最小距离d为半径的圆与切割维度相交，说明最近距离有可能还在相邻的子区域，此时点在左域，那么就需要遍历右域的点 ），
                   如果l不小于d那么完全没有比较进入相邻子域，这样也就减少了比较的次数。按照上面方法一直回溯完栈里面的节点。 

KD树在低纬空间中的效率比较高，但是当数据维度很高的时候查找效率就会下降，原因是在回溯的步骤中，以搜索点为中心，当时的最小距离为半径的
超球面会与该节点的与其相邻的子域相交，因此就需要在相邻的子域里面进行查询比较，随着维度的增高，回溯的时候相交会增多，查询比较也就增多
严重影响着查询效率，那么要提高查询效率，自然而然的办法就是减少查询的次数啊，那么怎么减少呢，人为设定一个回溯的次数（或者时间上线），
也就是以准确度来换取时间，这里就需要维护一个存储路径点优先级的队列，优先级怎么定义呢，就是按照搜索点到该节点分割维度的面上的距离大小
从小到大排序，这样当回溯的时候我们基本上就先考虑了离搜索点比较近的点，直到达到条件上限，这样得到的是一个近似值。上面的方法就是BBF算法
（best-bin-first）
'''

import numpy as np

class KD_node:
    
    def __init__(self,point=None,split=None,LL=None,RR=None):
        self.point = point
        self.split = split
        self.left = LL
        self.right = RR
        
def get_split(X):
    
    mean_v =np.mean(X,axis = 0) 
    val = sum((X-mean_v)**2)/float(X.shape[0])
    idxs = val.argsort()
    return idxs[-1]

def create_kdtree(datas):
    
    N = datas.shape[0]
    if N == 0:
        return

    ##计算各个维度的方差，以最大方差的维度作为切分维度    
    split = get_split(datas)
    ##在切分维度上以中位数的点为切分点
    split_sort = datas[:,split].argsort()
    point = datas[split_sort[N/2],:]
    root = KD_node(point, split)
    root.left = create_kdtree(datas[split_sort[:N/2]])
    root.right = create_kdtree(datas[split_sort[N/2+1:]])
    return root

def compute_dist(X1,X2):
    
    return np.sqrt(sum((X1-X2)**2))
   
def findNN(root,query):
    
    NN = root.point
    min_dist = compute_dist(query,NN)
    node_list = []
    temp_root = root 
    ##二分查找建立路径
    while temp_root:
        node_list.append(temp_root)
        dd = compute_dist(query,temp_root.point)
        if min_dist > dd:
            NN = temp_root.point
            min_dist = dd
        ss = temp_root.split
        if query[ss] <= temp_root.point[ss]:
            temp_root = temp_root.left
        else:
            temp_root = temp_root.right
    ##回溯查找
    while node_list:
        back_point = node_list.pop()
        ss = back_point.split
        ##判断是否需要进入父亲节点的子空间进行搜索 
        if abs(query[ss] - back_point.point[ss]) < min_dist:
            if query[ss] <= back_point.point[ss]:
                temp_root = back_point.right
            else:
                temp_root = back_point.left
            
            if temp_root:
                node_list.append(temp_root)
                cur_dist = compute_dist(query,temp_root.point)
                if min_dist > cur_dist:
                    min_dist = cur_dist
                    NN = temp_root.point  
    return NN, min_dist  

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

if __name__ == "__main__":
    
    filename = "E:/data/datingTestSet2.txt"
    datas,labels = load_data(filename)
    norm_datas = auto_norm(datas)
    test = norm_datas[0]
    root = create_kdtree(norm_datas[1:])
    NN, min_dist = findNN(root,test)
    print NN,min_dist
    
    
