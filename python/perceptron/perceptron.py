#-*- coding:utf-8 -*-

'''
参考李航《统计学习方法》第二章感知机.

   感知机是最简单的分类算法，对于线性不可分的数据，感知机会陷入无线循环，无法收敛，因此需要保证数据线性可分。
   课后习题2.1，感知机无法表示异或（XOR）问题，所谓异或问题就是全真或者全假为假，一真一假为真，用图表示明显看出来是线性不可分的：
   
                                     |
                              O      |         X
                                     |
                                     |
                      ---------------|----------------------
                                     |
                              X      |          O
                                     |
   
   感知机存在无数组解，这些解即依赖于初值的选择，也依赖于误分类点的选择顺序，为了得到唯一的超平面，需要对分离超平面增加约束条件，即就是svm的做法。
   
'''

import numpy as np

class Perceptron:

    def __init__(self,eta0,w0,b0):

        self.eta = eta0
        self.w = w0
        self.b = b0

    def loss(self,x,y):
        ret = -y*(np.dot(self.w,x)+self.b)
        return ret

    def adjust(self,x,y):

        self.w += self.eta*y*x
        self.b += self.eta*y

    def train(self,X,Y):
        
        iter = 0
        N = Y.shape[0]
        flag = True

        while flag:
            for i in range(N):
                loss = self.loss(X[i],Y[i])
                if -loss <= 0:
                    iter += 1
                    self.adjust(X[i],Y[i])
                    print "%d iter ....."%(iter)
                    flag = True
                    break
                else:
                    flag = False

        print "*****w*****"
        print self.w
        print "b : %f"%(self.b)

class Perceptron_dual:

    def __init__(self,eta0,alpha0,b0):
  
        self.eta = eta0
        self.alpha = alpha0
        self.b = b0

    @staticmethod
    def gram(X):
    
        return np.dot(X,X.T)

    def train(self,X,Y):

        N = Y.shape[0] 
        flag = True
        iter = 0

        while(flag):
            for i in range(N):
                g = self.gram(X)
                if Y[i]*(np.sum(self.alpha*Y*g[i])+self.b)<= 0:
                    iter += 1
                    print "%d iter...."%(iter)
                    self.alpha[i] +=  self.eta
                    self.b += self.eta*Y[i]
                    flag = True
                    break
                else:
                    flag = False
        w = np.dot(X.T,np.multiply(self.alpha,Y))
        print "***** w *****"
        print w
        print "b : %f"%(self.b)

if __name__ == "__main__":

    X = np.array([[3,3],[4,3],[1,1]])
    
    Y = np.array([1,1,-1])

    p = Perceptron(1,np.zeros(X.shape[1]),0)
    p.train(X,Y)

    p_dual = Perceptron_dual(1,np.zeros(Y.shape[0]),0)
    p_dual.train(X,Y)

