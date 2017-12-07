#-*- coding:utf-8 -*-

'''
Created on 2016年10月28日

@author: wangzhe
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import random

#加载数据，顺便把偏置项加入到特征中
def load_data(path):
    
    datas = []
    labels = []

    fr = open(path)
    for line in fr:
        items = line.strip().split("\t")
        datas.append(map(float,[items[0],items[1],1.0]))
        labels.append(int(items[2]))
    return datas,labels

class Lr():
    
    def __init__(self,datas,labels,max_iter,rate):
        
        self.datas = datas
        self.labels = labels
        self.iter = max_iter
        self.rate = rate
#         self.weights = np.ones((len(datas[0]),1))
        self.weights = None
        self.history_w = []
    
    @staticmethod
    def sigmoid(x):
        return 1.0/(1.0+np.exp(-x))
    
    def batch_grad_descent(self):
        
        self.weights = np.ones((len(self.datas[0]),1))
        data_matrix = np.mat(self.datas)
        label_matrix = np.mat(self.labels).transpose()
        
        for i in range(1000):
            if i == self.iter:
                break
            i += 1
            h = self.sigmoid(data_matrix*self.weights)
            diff = h - label_matrix
            self.weights = self.weights - self.rate * data_matrix.transpose()*diff
            self.history_w.append(self.weights)
    
    def stochastic_gradient_descent(self):
        
        data_arr = np.array(self.datas)
        m,n = np.shape(data_arr)
        self.weights = np.ones(n)
        for i in range(m):
            h = self.sigmoid(sum(data_arr[i]*self.weights))  #挑选（伪随机）第i个实例来更新权值向量
            error = self.labels[i] - h
            self.weights = self.weights + data_arr[i] * self.rate * error
            self.history_w.append(self.weights)
    
    def update_stochastic_gradtent_descent(self):
        
        data_arr = np.array(self.datas)
        m,n = np.shape(data_arr)
        self.weights = np.ones(n)                
        
        for j in range(self.iter):
            for i in range(m):
                alpha = 4/(1.0+j+i)+0.0001
                rand_index = int(random.uniform(0,m))
                h = self.sigmoid(sum(data_arr[rand_index]*self.weights))
                error = self.labels[rand_index] - h
                self.weights = self.weights + data_arr[rand_index] * alpha * error
                self.history_w.append(self.weights)
             
    def plot_gif(self):
       
        fig = plt.figure()
        ax = fig.add_subplot(111)
        line, = ax.plot([], [], 'b', lw=2)
       
        def draw_line(weights):
            x = np.arange(-5.0, 5.0, 0.1)
            y = (-weights[-1]-weights[0]*x)/weights[1]
            line.set_data(x, y)
            return line,
       
        def init():
            
            data_arr = np.array(self.datas)
            labels_arr = np.array(self.labels)
            
            pos_idx = labels_arr == 1
            xcord1 = data_arr[pos_idx][:,0]
            ycord1 = data_arr[pos_idx][:,1]
            xcord2 = data_arr[~pos_idx][:,0]
            ycord2 = data_arr[~pos_idx][:,1]
            
            ax = fig.add_subplot(111)
            ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
            ax.scatter(xcord2, ycord2, s=30, c='green')
            plt.xlabel('X1'); plt.ylabel('X2');
           
            return draw_line(np.zeros((data_arr.shape[1],1)))
       
        def animate(i):
            return draw_line(self.history_w[i])
       
        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(self.history_w), interval=10, repeat=False,
                                       blit=True)
        plt.show()
#         anim.save('gradAscent.gif', fps=2, writer='imagemagick')
        

if __name__ == "__main__":
    
    path = "./../../data/testSet.txt"
    datas,labels = load_data(path)
    lr = Lr(datas,labels,1000,0.001)
    lr.batch_grad_descent()
    lr.plot_gif()
    
    
