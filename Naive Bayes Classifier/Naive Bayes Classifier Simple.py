# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 17:04:54 2024

@author: nucle
"""
############################## INCLUSIONS AND HEADERS #########################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

############## Marginal Probability calculator #####
############## given the data to search ############
############# calculates the marginal probability

def marginal_prob (data, obj):
    ct=0
    if len(data)==0 : ret =0
    else: 
        ct = 0
        #print("HERE : searching for :",obj)
        for i in range(len(data)):
           if (data[i]== obj):ct+=1
        ret = ct/len(data)
        
    return ret
            
######### create a classifier from the data ################

def classifier(data, new, ctgry):
    l = len(data)
    n = len(data.columns)
    #print("l is : ",l, " n is :",n)
    if l == 0 : ret = 0
    
    else:
        classif = []
        for j in range(len(ctgry)):
            msk = np.zeros(l)
            for i in range(l):
                if (data.iloc[i,-1]==ctgry[j]): msk[i]=1
            msk = msk.astype(bool)
            temp = data[msk]
            p = len(temp)/l
            for i in range(n-1):
                ut = np.array(temp.iloc[:,i])
                t = marginal_prob(ut,new[i])
                p*=t
            classif.append(p)
        
        pos  = 0
        for i in range(len(ctgry)):
            if (classif[i]> classif[pos]): pos = i
        ret = ctgry[pos]
        
    return ret

###################### Read File and Preliminary Analysis #####################

df = pd.read_csv("C:/Users/nucle/IMPORTANT SCRIPTS/Naive Bayes Simple 1.csv")
#print(df)                                             # mainly for testing
df=df.drop(['Unnamed: 0'],axis=1)
#print(df)                                             # mainly for testing

n = len(df.columns) 
#print("n is : ", n)
X = df.iloc[:,:-1]                                     # split data 
#print(X)
Y = df.iloc[:,n-1]                                     # split output classified
#print(Y)
#'''

ctgry = np.unique(Y)          
print(ctgry)

pts = ['C','D','A','Medium','A','+']                  # categorize given point

ret = classifier(df,pts,ctgry)

print(ret)
