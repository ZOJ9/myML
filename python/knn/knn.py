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

    '''
    此方法是对所有的列进行归一化处理
    1. numpy.ndarray.min: 没有参数返回所有的最小值，axis=0返回每列的最小值，axis=1 返回每行的最小值
    2.numpy.tile([0,0],(2,1))#在列方向上重复[0,0]1次，行2次 ；numpy.tile([0,0],5)#在列方向上重复[0,0]5次，默认行1次
    '''

    min_v = datas.min(0)
    max_v = datas.max(0)
#     norm_datas = np.zeros(datas.shape)
    norm_datas = datas - np.tile(min_v,(datas.shape[0],1))
    norm_datas = norm_datas/np.tile(max_v-min_v,(datas.shape[0],1))
    return norm_datas


def classify0(test, datas, labels, k):

    '''
    :param test: 测试集，这里是一行数据
    :param datas: 训练集
    :param labels: 标签
    :param k:
    求具体一行数据的k近邻数据，确定其标签：先求这行数据到每行数据的距离，然后选出k近邻数据，投票表决其标签。
    距离sqrt((x1-y1)^2+(x2-y2)^2+(x3-y3)^3)
    1. np.array.argsort() 返回的是数组值从小到大的索引值,axis=0按列排序，axis=1 按照行排序，数据加-号是反向排序；
    2. python 字典排序：
        #按字典值排序（默认为升序）如果要降序排序,可以指定reverse=True
        sorted_x = sorted(x.iteritems(), key=operator.itemgetter(1))
        #取代方法是,用lambda表达式
        sorted_x = sorted(x.iteritems(), key=lambda x : x[1])

    '''
    
    m = datas.shape[0]
    diff_mat = np.tile(test,(m,1))-datas
    sqrt_mat = diff_mat**2
    dist_mat = sqrt_mat.sum(axis=1) #axis按照行求和
    dist_mat = dist_mat**0.5
    dist_mat = dist_mat.argsort()

    class_count = {}
    for i in range(k):
        vote_label = labels[dist_mat[i]]
        class_count[vote_label] = class_count.get(vote_label,0)+1 #投票决定标签。
    sort_class = sorted(class_count.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sort_class[0][0]


def train(datas, labels, k):
    
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
    datas, labels = load_data(filename)
    norm_datas = auto_norm(datas)
    train(norm_datas, labels, 3)
