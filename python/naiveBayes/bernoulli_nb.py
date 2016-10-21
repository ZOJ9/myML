#-*- coding:utf-8 -*-
'''
Created on 2016年10月20日

@author: wangzhe

info: 伯努利分布关心的是这个词是否在某个类下面出现，而不关心具体出现几次（例如掷硬币，我们关心的不是
               硬币正面出现几次的概率而是关心的硬币正面出现的概率），因此如果数据服从伯努利分布，我们就以文档为
            粒度，不但要考虑词在某个类下面出现的概率，还需要考虑词不出现在某个类下面的概率。
参考：http://nlp.stanford.edu/IR-book/html/htmledition/the-bernoulli-model-1.html
'''

import re
import os
import math
import random
import numpy as np
from collections import Counter

def proce_data(folder):
    
    ham_folder = folder+"ham/"
    spam_folder = folder+"spam/"
    
    tr_datas = []
    tt_datas = []
    
    ham_files = os.listdir(ham_folder) 
    for filename in ham_files:
        lst = []
        with open(ham_folder+filename) as fr:
            for line in fr:
                tokens_lst = re.split(r'\W*',line.strip())
                if len(tokens_lst)==0:
                    continue
                tokens_lst = [tok.lower() for tok in tokens_lst if len(tok) > 2]
                lst += tokens_lst
        if random.randint(1,100) <=80:
            tr_datas.append(lst)
        else:
            tt_datas.append(lst)
    tr_labels = [1]*len(tr_datas)
    tt_labels = [1]*len(tt_datas)
                
    spam_files = os.listdir(spam_folder)
    for filename in spam_files:
        lst = []
        with open(spam_folder+filename) as fr :
            for line in fr:
                tokens_lst = re.split(r'\W*',line.strip())
                if len(tokens_lst)==0:
                    continue
                tokens_lst = [tok.lower() for tok in tokens_lst if len(tok) > 2]
                lst += tokens_lst
        if random.randint(1,100) <=80:
            tr_datas.append(tokens_lst)
            tr_labels.append(-1)
        else:
            tt_datas.append(tokens_lst)
            tt_labels.append(-1)
    print "train data len : %d"%(len(tr_labels))
    print "test data len : %d"%(len(tt_labels))
    return [tr_labels,tr_datas],[tt_labels,tt_datas]

def bnuli_nb(trains,smoot=1.0):
    
    tr_labels = np.array(trains[0])
    tr_datas = np.array(trains[1])
    #0.计算先验概率
    y_count_dict = Counter(tr_labels)
    y_prior_dict = dict((k,float(v)/len(tr_labels))
                        for k,v in y_count_dict.items())
    
    # 以伯努利分布计算类下词的条件概率
    c_words_doc_dict = {}
    for cate in set(tr_labels):
        #1.所有的词
        words_all = set(reduce(lambda x,y:list(x)+list(y),list(tr_datas)))
        #2.分类查找词
        cidx = tr_labels==cate
        c_words = tr_datas[cidx]
        #3.计算类下词的条件概率
        N_d = len(c_words)
        c_words = map(lambda x:list(set(x)),list(c_words))
        c_words = reduce(lambda x,y:x+y,c_words)
        c_words_count_dict = Counter(c_words)
        c_words_doc_dict[cate] = dict((k,(v+float(smoot))/float(N_d+2.0))
                                for k,v in c_words_count_dict.items())
        #4.计算类下面没有出现的词的概率
        c_diff_words = list(words_all.difference(set(c_words)))
        diff_prob = (float(smoot)+0.0)/(N_d+2.0)
        c_diff_count_dict = dict([(k,diff_prob) for k in c_diff_words])
        #5.合并字典
        c_words_doc_dict[cate].update(c_diff_count_dict)      
    return y_prior_dict,c_words_doc_dict

def bernoulli_pred(y_prior_dict,c_words_doc_dict,tests,smooth=1.0):
    
    pred_cate_list = []
    tt_datas = tests[1]
    
    for items in tt_datas:
        pred_cate_dict = {}
        for c in y_prior_dict.keys():
            score_c = math.log(y_prior_dict[c])
            for k,v in c_words_doc_dict[c].iteritems():
                if k in items:
                    score_c += math.log(v)
                else:
                    score_c += math.log(1-v)
            pred_cate_dict[c] = float(score_c)
        cate_range = sorted(pred_cate_dict.iteritems(), key=lambda x:x[1], reverse=True)
        pred_cate = cate_range[0][0]
        pred_cate_list.append(pred_cate)
    print "predict file len : %d"%len(pred_cate_list)
    return pred_cate_list

def evaluation(src_cate_list,pred_cate_list) :
    
    print "*************"
    print src_cate_list
    print pred_cate_list

    src_cate_set = set(src_cate_list)
    s = 0
    for cate in src_cate_set :
        tp = 0
        fp = 0
        fn = 0
        for i in range(len(src_cate_list)) :
            if src_cate_list[i] == cate and pred_cate_list[i] == cate :
                tp += 1
            elif pred_cate_list[i] == cate :
                fp += 1
            elif src_cate_list[i] == cate :
                fn += 1
        s += tp
        s += fp
        s += fn
        if tp != 0 and fp != 0 :
            precision = tp/float(tp + fp)
            recall = tp/float(tp + fn)
            fval = precision*recall*2/(precision + recall)
        else :
            precision = 0.0
            recall = 0.0
            fval = 0.0 
        print ("%s"%cate).center(50,"*")
        print "precision : %f"%precision
        print "recall : %f"%recall
        print "F : %f"%fval

if __name__ == "__main__":
    
    trs = [["Chinese","Beijing","Chinese"],
              ["Chinese","Chinese","Shanghai"],
              ["Chinese","Macao"],
              ["Tokyo","Japan","Chinese"]]
    tr_labels = [1,1,1,-1]
    tts = [["Chinese","Chinese","Chinese","Tokyo","Japan"]]
    y_prior_dict,c_words_doc_dict = bnuli_nb([tr_labels,trs])
    pred_cate_list = bernoulli_pred(y_prior_dict,c_words_doc_dict,[0,tts])
#     folder = "e:/data/email/"
#     trains,tests = proce_data(folder)
#     y_prior_dict,c_words_doc_dict = bnuli_nb(trains)
#     pred_cate_list = bernoulli_pred(y_prior_dict,c_words_doc_dict,tests)
#     evaluation(tests[0],pred_cate_list)

