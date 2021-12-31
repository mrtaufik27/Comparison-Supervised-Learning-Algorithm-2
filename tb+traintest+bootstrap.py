# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 11:15:05 2018

@author: Taufik : training and testing and bootstrap
"""

#Import required libraries
import os
import pandas as pd
import statsmodels.api as sm
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score
from patsy import dmatrices
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from scipy import interp
from itertools import cycle
import random

os.chdir('C:/Users/USER/Dropbox/thesis berkah/command')

# Read data
#df = pd.read_csv("verbalAutopsy.csv")
df = pd.read_csv("va.csv")

# Summarize
df.info()
df.describe().transpose()
df.shape
df.head(5)
df['tb'].value_counts(ascending=True)
# pd.crosstab(df['x1'], df['tb'])

df['pro'] = df['pro'].astype(object)
df['ghos'] = df['ghos'].astype(object)
df['agSx'] = df['agSx'].astype(object)

# Bootstrap resampling
def bootstrap_resample(X, n=None):
    """ Bootstrap resample an array_like
    Parameters
    ----------
    X : array_like
      data to resample
    n : int, optional
      length of resampled array, equal to len(X) if n==None
    Results
    -------
    returns X_resamples
    """
    if isinstance(X, pd.Series):
        X = X.copy()
        X.index = range(len(X.index))
    if n == None:
        n = len(X)
        
    resample_i = np.floor(np.random.rand(n)*len(X)).astype(int)
    X_resample = np.array(X[resample_i])  # TODO: write a test demonstrating why array() is important
    return X_resample


df_resampled= bootstrap_resample(df.index, 18990)
df.new =  pd.DataFrame(df_resampled)
#fig = plt.figure(1, figsize=(6,5), dpi=80)
#ax1 = fig.add_subplot(111)
#va.new.hist(ax=ax1)
#ax1.set_xlabel('BigSampIDs', **axis_font)
#ax1.set_ylabel('Frequency', **axis_font)
#ax1.set_xlabel('Histogram of BigSampIds', **axis_font)

df.new1 = pd.DataFrame(df.new)
df1 = df.loc[df.index ==df.new1.loc[0,0]]
for i in range(18989):
    df2 = df.loc[df.index == df.new1.loc[i+1,0]]
    df1 = df1.append(df2)

#Category
y, X = dmatrices('tb ~ pro+ghos+agSx+DRgrp', df1, return_type = 'dataframe')

#Spliting dataset into training and testing
random.seed(12321)
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#These parameters will give 70 % to training, and 15 % each to test and val sets. Hope this 

#------------reindexing
x_train = x_train.reset_index()
del x_train['index']

x_test = x_test.reset_index()
del x_test['index']

y_train = y_train.reset_index()
del y_train['index']

y_test = y_test.reset_index()
del y_test['index']

y_test['tb'].value_counts()
y_train['tb'].value_counts()

#--------------Build Logistic Regression Model-----------------------
#Fit Logit model for training and testing set           
logit = sm.Logit(y_train,x_train)
result = logit.fit(method='bfgs')
result.summary()

#Make predictions
predictions_lr = result.predict(x_test) 
pp1 = pd.DataFrame(predictions_lr)
for i in range(5697):
 if pp1[0][i] > 0.1:
    pp1[0][i] = 1
 else:
    pp1[0][i] = 0
tb1 = pd.crosstab(pp1[0], y_test['tb'])
ac1 = accuracy_score([ 1 if p > 0.5 else 0 for p in predictions_lr ], y_test)

# AUC
predictions_lr = result.predict(x_test) 
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predictions_lr)
auc(false_positive_rate, true_positive_rate)


#--------------------------Decision Tree----------------------------
model_dt = DecisionTreeClassifier()
model_dt.fit(x_train, y_train)

#Make predictions
predictions_dt = model_dt.predict_proba(x_test)
pp2 = pd.DataFrame(predictions_dt[:,1])
for i in range(5697):
 if pp2[0][i] > 0.5:
    pp2[0][i] = 1
 else:
    pp2[0][i] = 0
tb2 = pd.crosstab(pp2[0], y_test['tb'])
ac2 = accuracy_score([ 1 if p > 0.5 else 0 for p in  predictions_dt[:,1] ], y_test)

# AUC
predictions_dt = model_dt.predict_proba(x_test)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predictions_dt[:,1])
auc(false_positive_rate, true_positive_rate)


#----------------------Random Forest------------------------------
model_rf = RandomForestClassifier(n_estimators=1000)
model_rf.fit(x_train,y_train)

#Make predictions
predictions_rf = model_rf.predict_proba(x_test)
pp3 = pd.DataFrame(predictions_rf[:,1])
for i in range(5697):
 if pp3[0][i] > 0.5:
    pp3[0][i] = 1
 else:
    pp3[0][i] = 0
tb3 = pd.crosstab(pp3[0], y_test['tb'])
ac3 = accuracy_score([ 1 if p > 0.5 else 0 for p in  predictions_rf[:,1] ], y_test)

# AUC
predictions_rf = model_rf.predict_proba(x_test)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predictions_rf[:,1])
auc(false_positive_rate, true_positive_rate)

#-------------------------------Neural Network---------------------------    
model_nn = MLPClassifier()
model_nn.fit(x_train,y_train)

#Make predictions
predictions_nn = model_nn.predict_proba(x_test)
pp4 = pd.DataFrame(predictions_nn[:,1])
for i in range(5697):
 if pp4[0][i] > 0.5:
    pp4[0][i] = 1
 else:
    pp4[0][i] = 0     
tb4 = pd.crosstab(pp4[0], y_test['tb'])
ac4 = accuracy_score([ 1 if p > 0.28 else 0 for p in  predictions_nn[:,1] ], y_test['tb'])   

# AUC
predictions_nn = model_nn.predict_proba(x_test)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predictions_nn[:,1])
auc(false_positive_rate, true_positive_rate)

#-------------------------Plots 4 models---------------------------
predictions_lr = result.predict(x_test)
predictions_dt = model_dt.predict_proba(x_test)
predictions_rf = model_rf.predict_proba(x_test)
predictions_nn = model_nn.predict_proba(x_test)
ymin = min(y_test['tb']); ymax = max(y_test['tb'])
ylm = [ymin,ymax]

fig = plt.figure(figsize=(4.33, 3.75), dpi = 250)
axis_font = {'fontname':'Times New Roman', 'size':'12'}
ax1 = fig.add_subplot(2, 2, 1)
plt.scatter(predictions_lr, y_test['tb'],  edgecolors='blue', facecolors='none', linewidths=1, s=10 )
plt.plot([0.5,0.5],[-0.1,1.1], color='red')
#ax1.set_xlabel('Estimated Risk', **axis_font)
ax1.set_ylabel('Died', rotation=0, **axis_font)
ax1.yaxis.set_label_coords(0,1)
ax1.text(ymax-0.25,1.015, str(x_test.shape[0])+' obs', 
         transform=ax1.transAxes, **axis_font)
plt.ylim(-0.1,1.1)
plt.xlim(-0.01,1.01)
#labs = [0.00,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.00]
#labels = ['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0']
labs = [0.00,0.20,0.40,0.60,0.80,1.00]
labels = ['0.0','0.2','0.4','0.6','0.8','1.0']
plt.xticks(labs, labels, fontsize=8)
ttl = plt.title('Logistic Regression', fontsize=12, **axis_font) #fontweight="bold", 
ttl.set_position([.5, 1.05])
ylab = ['No', 'Yes']
yn = [0,1]
plt.yticks(yn, ylab, **axis_font)
ax1.text(0.3, 0.7, str(tb1[1][0])+'-',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax1.transAxes, **axis_font, color='red')
ax1.text(0.9, 0.7, tb1[1][1],
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax1.transAxes, **axis_font)
ax1.text(0.3, 0.15, tb1[0][0],
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax1.transAxes, **axis_font)
ax1.text(0.9, 0.15, str(tb1[0][1])+'+',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax1.transAxes, **axis_font, color='red')
bbox_props = dict(boxstyle='square', facecolor='ivory', lw=0.5)
ax1.text(0.2, 0.45, 'Accuracy: '+str(round(ac1*100,2))+'%', **axis_font, 
        bbox=bbox_props)

ax2 = fig.add_subplot(2, 2, 2)
plt.scatter(predictions_dt[:,1], y_test['tb'],  edgecolors='blue', facecolors='none', linewidths=1, s=10)
plt.plot([0.5,0.5],[-0.1,1.1], color='red')
plt.ylim(-0.1,1.1)
plt.xlim(-0.01,1.01)
#ax2.set_xlabel('Estimated Risk', **axis_font)
ax2.set_ylabel('Died', rotation=0, **axis_font)
ax2.yaxis.set_label_coords(0,1)
ax2.text(ymax-0.25,1.015, str(x_test.shape[0])+' obs', 
         transform=ax2.transAxes, **axis_font)
plt.xticks(labs, labels, fontsize=8)
ttl = plt.title('Decision Tree', fontsize=12, **axis_font) #fontweight="bold",
ttl.set_position([.5, 1.05])
ylab = ['No', 'Yes']
yn = [0,1]
plt.yticks(yn, ylab, **axis_font)
ax2.text(0.3, 0.7, str(tb2[1][0])+'-',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax2.transAxes, **axis_font, color='red')
ax2.text(0.9, 0.7, tb2[1][1],
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax2.transAxes, **axis_font)
ax2.text(0.3, 0.15, tb2[0][0],
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax2.transAxes, **axis_font)
ax2.text(0.9, 0.15, str(tb2[0][1])+'+',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax2.transAxes, **axis_font, color='red')
ax2.text(0.2, 0.45, 'Accuracy: '+str(round(ac2*100,2))+'%', **axis_font, 
        bbox=bbox_props)

ax3 = fig.add_subplot(2, 2, 3)
plt.scatter(predictions_rf[:,1], y_test['tb'],  edgecolors='blue', facecolors='none', linewidths=1, s=10)
plt.plot([0.5,0.5],[-0.1,1.1], color='red')
plt.ylim(-0.1,1.1)
plt.xlim(-0.01,1.01)
#ax3.set_xlabel('Estimated Risk', **axis_font)
ax3.set_ylabel('Died', rotation=0, **axis_font)
ax3.yaxis.set_label_coords(0,1)
ax3.text(ymax-0.25,1.015, str(x_test.shape[0])+' obs', 
         transform=ax3.transAxes, **axis_font)
plt.xticks(labs, labels, fontsize=8)
ttl = plt.title('Random Forest', fontsize=12, **axis_font) #,fontweight="bold"
ttl.set_position([.5, 1.05])
ylab = ['No', 'Yes']
yn = [0,1]
plt.yticks(yn, ylab, **axis_font)
ax3.text(0.3, 0.7, str(tb3[1][0])+'-',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax3.transAxes, **axis_font, color='red')
ax3.text(0.9, 0.7, tb3[1][1],
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax3.transAxes, **axis_font)
ax3.text(0.3, 0.15, tb3[0][0],
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax3.transAxes, **axis_font)
ax3.text(0.9, 0.15, str(tb3[0][1])+'+',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax3.transAxes, **axis_font, color='red')
ax3.text(0.2, 0.45, 'Accuracy: '+str(round(ac3*100,2))+'%', **axis_font, 
        bbox=bbox_props)

ax4 = fig.add_subplot(2, 2, 4)
plt.scatter(predictions_nn[:,1], y_test['tb'],  edgecolors='blue', facecolors='none', linewidths=1, s=10)
plt.plot([0.5,0.5],[-0.1,1.1], color='red')
plt.ylim(-0.1,1.1)
plt.xlim(-0.01,1.01)
#ax4.set_xlabel('Estimated Risk', **axis_font)
ax4.set_ylabel('Died', rotation=0, **axis_font)
ax4.yaxis.set_label_coords(0,1)
ax4.text(ymax-0.25,1.015, str(x_test.shape[0])+' obs', 
         transform=ax4.transAxes, **axis_font)
plt.xticks(labs, labels, fontsize=8)
ttl = plt.title('Neural Network', fontsize=12, **axis_font) #,fontweight="bold"
ttl.set_position([.5, 1.05])
ylab = ['No', 'Yes']
yn = [0,1]
plt.yticks(yn, ylab, **axis_font)
ax4.text(0.3, 0.7, str(tb4[1][0])+'-',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax4.transAxes, **axis_font, color='red')
ax4.text(0.9, 0.7, tb4[1][1],
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax4.transAxes, **axis_font)
ax4.text(0.3, 0.15, tb4[0][0],
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax4.transAxes, **axis_font)
ax4.text(0.9, 0.15, str(tb4[0][1])+'+',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax4.transAxes, **axis_font, color='red')
ax4.text(0.2, 0.45, 'Accuracy: '+str(round(ac4*100,2))+'%', **axis_font, 
        bbox=bbox_props)
plt.tight_layout()
plt.subplots_adjust(top=0.92,
bottom=0.074,
left=0.089,
right=0.955,
hspace=0.5,
wspace=0.31)

#--------------------Compute ROC curve and ROC area for each class-------------
fpr = dict()
tpr = dict()
roc_auc = dict()

fpr[0], tpr[0], _ = roc_curve(y_test, predictions_lr)
roc_auc[0] = auc(fpr[0], tpr[0])
fpr[1], tpr[1], _ = roc_curve(y_test, predictions_dt[:,1])
roc_auc[1] = auc(fpr[1], tpr[1])
fpr[2], tpr[2], _ = roc_curve(y_test, predictions_rf[:,1])
roc_auc[2] = auc(fpr[2], tpr[2])
fpr[3], tpr[3], _ = roc_curve(y_test, predictions_nn[:,1])
roc_auc[3] = auc(fpr[3], tpr[3])

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(4)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(4):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

#------------------------------------------------------------------------
# Plot all ROC curves
fig = plt.figure(figsize=(4.33, 3.75), dpi = 250)
axis_font = {'fontname':'Times New Roman', 'size':'12'}
lw = 1
mod_name = ['LR', 'DT', 'RF','NT']
colors = cycle(['aqua', 'darkorange', 'cornflowerblue','deeppink'])
for i, color in zip(range(4), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='AUROC of {0} (area = {1:0.2f})'
             ''.format(mod_name[i], roc_auc[i]))

plt.yticks(**axis_font)
plt.xticks(**axis_font)
plt.plot([0, 1], [0, 1], lw=lw, color='black')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('False positive rate', **axis_font)
ax1.set_ylabel('Sensitivity', rotation=0, **axis_font)
ax1.yaxis.set_label_coords(0,1)
ax1.text(ymax-0.35,1.015, 'test sample: n='+str(y_test.shape[0]), 
         transform=ax1.transAxes, **axis_font)
#ttl = plt.title('ROC Curves: Bootstrap+Train:Test (70:30)',fontweight="bold", fontsize=18)
#ttl.set_position([.5, 1.05])
plt.legend(fontsize=12,loc="lower right",prop={'family':'Times New Roman'})
plt.subplots_adjust(
top=0.933,
bottom=0.127,
left=0.096,
right=0.924,
hspace=0.19,
wspace=0.2)

#------------------------------------------------------------------------

roc_auc = dict()
roc = pd.DataFrame()
acc = pd.DataFrame()

for j in range(100):
    #--------------Build Logistic Regression Model-----------------------
    #Fit Logit model for training and testing set           
    logit = sm.Logit(y_train,x_train)
    result = logit.fit(method='bfgs')
    result.summary()
    #Make predictions
    predictions_lr = result.predict(x_test) 
    pp1 = pd.DataFrame(predictions_lr)
    for i in range(5697):
     if pp1[0][i] > 0.1:
        pp1[0][i] = 1
     else:
        pp1[0][i] = 0
    tb1 = pd.crosstab(pp1[0], y_test['tb'])
    ac1 = accuracy_score([ 1 if p > 0.5 else 0 for p in predictions_lr ], y_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predictions_lr)
    auc(false_positive_rate, true_positive_rate)
    roc_auc[0] = auc(false_positive_rate, true_positive_rate)
    #--------------------------Decision Tree----------------------------
    model_dt = DecisionTreeClassifier()
    model_dt.fit(x_train, y_train)
    #Make predictions
    predictions_dt = model_dt.predict_proba(x_test)
    pp2 = pd.DataFrame(predictions_dt[:,1])
    for i in range(5697):
     if pp2[0][i] > 0.5:
        pp2[0][i] = 1
     else:
        pp2[0][i] = 0
    tb2 = pd.crosstab(pp2[0], y_test['tb'])
    ac2 = accuracy_score([ 1 if p > 0.5 else 0 for p in  predictions_dt[:,1] ], y_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predictions_dt[:,1])
    auc(false_positive_rate, true_positive_rate)
    roc_auc[1] = auc(false_positive_rate, true_positive_rate)
    #----------------------Random Forest------------------------------
    model_rf = RandomForestClassifier(n_estimators=1000)
    model_rf.fit(x_train,y_train)
    pp3 = pd.DataFrame(predictions_rf[:,1])
    for i in range(5697):
     if pp3[0][i] > 0.5:
        pp3[0][i] = 1
     else:
        pp3[0][i] = 0
    tb3 = pd.crosstab(pp3[0], y_test['tb'])
    ac3 = accuracy_score([ 1 if p > 0.5 else 0 for p in  predictions_rf[:,1] ], y_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predictions_rf[:,1])
    auc(false_positive_rate, true_positive_rate)
    roc_auc[2] = auc(false_positive_rate, true_positive_rate)
    #-------------------------------Neural Network---------------------------    
    model_nn = MLPClassifier()
    model_nn.fit(x_train,y_train)
    #Make predictions
    predictions_nn = model_nn.predict_proba(x_test)
    pp4 = pd.DataFrame(predictions_nn[:,1])
    for i in range(5697):
     if pp4[0][i] > 0.5:
        pp4[0][i] = 1
     else:
        pp4[0][i] = 0     
    tb4 = pd.crosstab(pp4[0], y_test['tb'])
    ac4 = accuracy_score([ 1 if p > 0.28 else 0 for p in  predictions_nn[:,1] ], y_test['tb'])   
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predictions_nn[:,1])
    auc(false_positive_rate, true_positive_rate)
    roc_auc[3] = auc(false_positive_rate, true_positive_rate)

    ac1 = str(round(ac1*100,2))    
    ac2 = str(round(ac2*100,2))    
    ac3 = str(round(ac3*100,2))
    ac4 = str(round(ac4*100,2))
    acc[j] = (ac1,ac2,ac3,ac4)
    roc[j] = (roc_auc[0],roc_auc[1],roc_auc[2],roc_auc[3])

acc.to_csv("accuracy bootstrap.csv", encoding='utf-8', index=False)
roc.to_csv("roc bootstrap.csv", encoding='utf-8', index=False)