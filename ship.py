# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 20:35:36 2017

@author: Administrator
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler


data1 = pd.DataFrame(sio.loadmat(r'c://kaggle project//rader//project1//ship1.mat')['range_profile_data1'])
data2 = pd.DataFrame(sio.loadmat(r'c://kaggle project//rader//project1//ship2.mat')['range_profile_data2'])
data3 = pd.DataFrame(sio.loadmat(r'c://kaggle project//rader//project1//ship3.mat')['range_profile_data3'])
data4 = pd.DataFrame(sio.loadmat(r'c://kaggle project//rader//project1//ship4.mat')['range_profile_data4'])
data5 = pd.DataFrame(sio.loadmat(r'c://kaggle project//rader//project1//ship5.mat')['range_profile_data5'])
data6 = pd.DataFrame(sio.loadmat(r'c://kaggle project//rader//project1//ship6.mat')['range_profile_data6'])
ave1 = sum(data1[:].mean())/len(data1[0:])
ave2 = sum(data2[:].mean())/len(data2[0:])
ave3 = sum(data3[:].mean())/len(data3[0:])
ave4 = sum(data4[:].mean())/len(data4[0:])
ave5 = sum(data5[:].mean())/len(data5[0:])
ave6 = sum(data6[:].mean())/len(data6[0:])
data1 = data1.replace(0,ave1)
data2 = data2.replace(0,ave2)
data3 = data3.replace(0,ave3)
data4 = data4.replace(0,ave4)
data5 = data5.replace(0,ave5)
data6 = data6.replace(0,ave6)

data1 = MinMaxScaler().fit_transform(data1.replace(0,ave1))
data2 = MinMaxScaler().fit_transform(data2.replace(0,ave1))
data3 = MinMaxScaler().fit_transform(data3.replace(0,ave1))
data4 = MinMaxScaler().fit_transform(data4.replace(0,ave1))
data5 = MinMaxScaler().fit_transform(data5.replace(0,ave1))
data6 = MinMaxScaler().fit_transform(data6.replace(0,ave1))

ship_data = np.hstack([data1,data2,data3,data4,data5,data6]) 

y1= np.ones((1,data1.shape[1]), dtype=np.int16 )
y2= 2*np.ones( (1,data2.shape[1]), dtype=np.int16 )
y3= 3*np.ones( (1,data3.shape[1]), dtype=np.int16 )
y4= 4*np.ones( (1,data4.shape[1]), dtype=np.int16 )
y5= 5*np.ones( (1,data5.shape[1]), dtype=np.int16 )
y6= 6*np.ones( (1,data6.shape[1]), dtype=np.int16 )

ship_target = np.hstack([y1,y2,y3,y4,y5,y6]) 


from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree   
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn import naive_bayes
from sklearn.metrics import confusion_matrix 
import time
import random
import sklearn
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm 
from matplotlib import axes

def TestSize(a):
    
    x_train,x_test,y_train,y_test = cross_validation.train_test_split(ship_data.T, ship_target.T, test_size = a, stratify = ship_target.T)
    Lsvc = LinearSVC()
    t1 = time.time()
    y_pred = Lsvc.fit(x_train,y_train).predict(x_test)
    t2 = time.time()
    Lsvc_matrix = confusion_matrix(y_test,y_pred)
    Lsvc_matrix = Lsvc_matrix.astype('float') / Lsvc_matrix.sum(axis=1)[:, np.newaxis]
    timediff1=t2-t1
    
    SGDC = SGDClassifier()
    t3 = time.time()
    y_pred = SGDC.fit(x_train,y_train).predict(x_test)
    t4 = time.time()
    SGDC_matrix = confusion_matrix(y_test,y_pred)
    SGDC_matrix = SGDC_matrix.astype('float') / SGDC_matrix.sum(axis=1)[:, np.newaxis]
    timediff2=t4-t3
    
    KNN = KNeighborsClassifier()  
    t5 = time.time()
    y_pred = KNN.fit(x_train,y_train).predict(x_test)
    t6 = time.time()
    KNN_matrix = confusion_matrix(y_test,y_pred)
    KNN_matrix = KNN_matrix.astype('float') / KNN_matrix.sum(axis=1)[:, np.newaxis] 
    timediff3=t6-t5
    
    RandomForest = RandomForestClassifier(n_estimators=20)  
    t7 = time.time()
    y_pred = RandomForest.fit(x_train,y_train).predict(x_test)  
    t8 = time.time()
    RF_matrix = confusion_matrix(y_test,y_pred)
    RF_matrix = RF_matrix.astype('float') / RF_matrix.sum(axis=1)[:,np.newaxis]
    timediff4=t8-t7
    
    DecisionTree= tree.DecisionTreeClassifier()  
    t9 = time.time()
    y_pred = DecisionTree.fit(x_train,y_train).predict(x_test)
    t10 = time.time()
    DT_matrix = confusion_matrix(y_test,y_pred)
    DT_matrix = DT_matrix.astype('float') / DT_matrix.sum(axis=1)[:,np.newaxis] 
    timediff5=t10-t9 
    
    LR = sklearn.linear_model.LogisticRegression()  
    t11 = time.time()
    y_pred = LR.fit(x_train,y_train).predict(x_test)
    t12 = time.time()
    LR_matrix = confusion_matrix(y_test,y_pred)
    LR_matrix = LR_matrix.astype('float') / LR_matrix.sum(axis=1)[:,np.newaxis]
    timediff6=t12-t11
    
    GradientBoosting = GradientBoostingClassifier(n_estimators=20)  
    t13 = time.time()
    y_pred = GradientBoosting.fit(x_train,y_train).predict(x_test) 
    t14 = time.time() 
    GB_matrix = confusion_matrix(y_test,y_pred)
    GB_matrix = GB_matrix.astype('float') / GB_matrix.sum(axis=1)[:,np.newaxis]
    timediff7=t14-t13
    
    Multi = sklearn.naive_bayes.MultinomialNB(alpha = 0.01)   
    t15 = time.time()
    y_pred = Multi.fit(x_train,y_train).predict(x_test)
    t16 = time.time()
    Multi_matrix = confusion_matrix(y_test,y_pred)
    Multi_matrix = Multi_matrix.astype('float') / Multi_matrix.sum(axis=1)[:,np.newaxis]
    timediff8=t16-t15
    
    accuracy_all = {'lsvc':Lsvc.score(x_test,y_test),'SGDC':SGDC.score(x_test,y_test),'KNN':KNN.score(x_test,y_test),'RandomForest':RandomForest.score(x_test,y_test),'DecisionTree':DecisionTree.score(x_test,y_test),'LogisticRegression':LR.score(x_test,y_test),'GradientBoosting':GradientBoosting.score(x_test,y_test),'MultinomialNB':Multi.score(x_test,y_test)}
    sorted(accuracy_all.keys(),reverse=1)
    time_all = {'lsvc':timediff1,'SGDC':timediff2,'KNN':timediff3,'RandomForest':timediff4,'DecisionTree':timediff5,'LogisticRegression':timediff6,'GradientBoosting':timediff7,'MultinomialNB':timediff8}
    sorted(time_all.keys(),reverse=1)
    Lsvc = np.diag(Lsvc_matrix) 
    GB = np.diag(GB_matrix)
    SGDC = np.diag(SGDC_matrix)
    KNN = np.diag(KNN_matrix)
    RandomForest = np.diag(RF_matrix)
    DecisionTree = np.diag(DT_matrix)
    LogisticRegression = np.diag(LR_matrix)
    MultinomialNB = np.diag(Multi_matrix)
   
    ship_all_accuracy = np.stack([Lsvc,GB,SGDC,KNN,RandomForest,DecisionTree,LogisticRegression,MultinomialNB],axis=1) 
    ship_all_time = np.stack([timediff1,timediff2,timediff3,timediff4,timediff5,timediff6,timediff7,timediff8],axis=0)
    accuracy_all_data = np.array(list(accuracy_all.values()))
    return(ship_all_accuracy,ship_all_time,accuracy_all_data)

def DrawDataAccuracy(accuracy_all_data,a):
    plt.bar([0,1,2,3,4,5,6,7],accuracy_all_data)
    plt.xticks([0,1,2,3,4,5,6,7],['lsvc','SGD','KNN','RF','DT','LR','GB','NB'])
    plt.title('training accuracy')
    plt.xlabel('type'+str(a))
    plt.ylabel('accuracy')
    
def DrawDataTime(ship_all_time,a):
    plt.bar([0,1,2,3,4,5,6,7],ship_all_time)
    plt.xticks([0,1,2,3,4,5,6,7],['lsvc','SGD','KNN','RF','DT','LR','GB','NB'])
    plt.title('training time')
    plt.xlabel('type'+str(a))
    plt.ylabel('time(ms)')

def draw_heatmap(data,xlabels,ylabels):
    cmap = cm.Blues   
    figure=plt.figure(facecolor='w')
    ax=figure.add_subplot(2,1,1,position=[0.1,0.15,0.8,0.8])
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels)
    vmax=data[0][0]
    vmin=data[0][0]
    for i in data:
        for j in i:
            if j>vmax:
                vmax=j
            if j<vmin:
                vmin=j
    map=ax.imshow(data,interpolation='nearest',cmap=cmap,aspect='auto',vmin=vmin,vmax=vmax)
    cb=plt.colorbar(mappable=map,cax=None,ax=None,shrink=0.5)
    plt.show()
    
xlabels=['lsvc','SGD','KNN','RF','DT','LR','GB','NB']
ylabels=['1','2','3','4','5','6']
ship_all_accuracy1,ship_all_time1,accuracy_all_data1 = TestSize(0.25)
DrawDataAccuracy(accuracy_all_data1,0.25)
DrawDataTime(ship_all_time1,0.25)
draw_heatmap(ship_all_accuracy1,xlabels,ylabels)

ship_all_accuracy2,ship_all_time2,accuracy_all_data2 = TestSize(0.3)
DrawDataAccuracy(accuracy_all_data2,0.3)
DrawDataTime(ship_all_time2,0.3)
draw_heatmap(ship_all_accuracy2,xlabels,ylabels)

ship_all_accuracy3,ship_all_time3,accuracy_all_data3 = TestSize(0.4)
DrawDataAccuracy(accuracy_all_data3,0.4)
DrawDataTime(ship_all_time3,0.4)
draw_heatmap(ship_all_accuracy3,xlabels,ylabels)

ship_all_accuracy4,ship_all_time4,accuracy_all_data4 = TestSize(0.5)
DrawDataAccuracy(accuracy_all_data4,0.5)
DrawDataTime(ship_all_time4,0.5)
draw_heatmap(ship_all_accuracy4,xlabels,ylabels)

'''
fig, axes = plt.subplots(2, 4,sharex = True,sharey = True)
for i in range(2):
    for j in range(2):
        for t in [0.25,0.3,0.4,0.5]:
            axes[i,j] = DrawDataAccuracy(eval('accuracy_all_data'+str()),t)
        
mpl.rcParams['font.size'] = 10 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for z in np.arange(ship_all_accuracy.shape[1]):
     xs = range(0,6)
     ys = ship_all_accuracy[:,z]
     'color =plt.cm.Set2(random.choice(range(plt.cm.Set2.N)))'
     ax.bar(xs, ys, zs=z, zdir='x', color=['yellow','blue','green','pink','red','grey','tan','brown'], alpha=0.8)
ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(xs))
ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(ys))
ax.set_zlabel('accuracy'+str(a))
plt.xticks([0,1,2,3,4,5],['ship1','ship2','ship3','ship4','ship5','ship6'],rotation = 60)
plt.yticks([0,1,2,3,4,5,6,7],['lsvc','SGD','KNN','RF','DT','LR','GB','NB'],rotation = 0)
plt.show()
'''
