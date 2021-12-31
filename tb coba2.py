# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 11:15:05 2018

@author: Taufik : whole dataset
"""

#Import required libraries
import os
import pandas as pd
import statsmodels.api as sm
import numpy as np
import pylab as pl
from sklearn.metrics import roc_curve, auc, accuracy_score
from patsy import dmatrices
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from scipy import interp
from itertools import cycle

os.chdir('C:/Users/USER/Dropbox/thesis berkah/command')

# Read data
df = pd.read_csv("va.csv")

# Summarize
df.info()
df.describe().transpose()
df.shape
df.head(5)

df.describe()
df.std()
# frequency table cutting presitge and whether or not someone was admitted
pd.crosstab(df['tb'], df['agSx'], rownames=['tb'])
# plot all of the columns
df.hist()
pl.show()

df['tb'].value_counts(ascending=True)
# pd.crosstab(df['x1'], df['tb'])

df['pro'] = df['pro'].astype(object)
df['ghos'] = df['ghos'].astype(object)
df['agSx'] = df['agSx'].astype(object)

#Category
y, X = dmatrices('tb ~ pro+ghos+agSx+DRgrp', df, return_type = 'dataframe')

#--------------Build Logistic Regression Model-----------------------
#Fit Logit model           
logit = sm.Logit(y,X)
result = logit.fit(metode='bfgs')
result.summary()

# look at the confidence interval of each coeffecient
result.conf_int()
# odds ratios only
np.exp(result.params)
# odds ratios and 95% CI
params = result.params
conf = result.conf_int()
conf['OR'] = params
conf.columns = ['2.5%', '97.5%', 'OR']
np.exp(conf)

#Make predictions
predictions_lr = result.predict(X) 
pp1 = pd.DataFrame(predictions_lr)
for i in range(9495):
 if pp1[0][i] > 0.5:
    pp1[0][i] = 1
 else:
    pp1[0][i] = 0
tb1 = pd.crosstab(pp1[0], y['tb'])
ac1 = accuracy_score([ 1 if p > 0.5 else 0 for p in predictions_lr ], y)

# AUC
predictions_lr = result.predict(X) 
false_positive_rate, true_positive_rate, thresholds = roc_curve(y, predictions_lr)
auc(false_positive_rate, true_positive_rate)


#--------------------------Decision Tree----------------------------
model_dt = DecisionTreeClassifier()
model_dt.fit(X, y)

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'

import graphviz 
class_names = [str(x) for x in model_dt.classes_]
feature_names = X.columns

dot_data = tree.export_graphviz(model_dt, out_file=None,
                         feature_names=feature_names, class_names=class_names,
                         filled=True, rounded=True,  
                         special_characters=True) 
graph = graphviz.Source(dot_data) 
graph.render('dt', view=True) 

model_dt = DecisionTreeClassifier(max_depth=3)
model_dt.fit(X, y)
class_names = [str(x) for x in model_dt.classes_]
feature_names = X.columns

dot_data = tree.export_graphviz(model_dt, out_file=None,
                         feature_names=feature_names, class_names=class_names,
                         filled=True, rounded=True,  
                         special_characters=True) 
graph = graphviz.Source(dot_data) 
graph.render('dt1', view=True) 

#Make predictions
predictions_dt = model_dt.predict_proba(X)
pp2 = pd.DataFrame(predictions_dt[:,1])
for i in range(9495):
 if pp2[0][i] > 0.5:
    pp2[0][i] = 1
 else:
    pp2[0][i] = 0
tb2 = pd.crosstab(pp2[0], y['tb'])
ac2 = accuracy_score([ 1 if p > 0.5 else 0 for p in  predictions_dt[:,1] ], y)

# AUC
predictions_dt = model_dt.predict_proba(X)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y, predictions_dt[:,1])
auc(false_positive_rate, true_positive_rate)


#----------------------Random Forest------------------------------
model_rf = RandomForestClassifier(n_estimators=1000)
model_rf.fit(X,y)

feature_imp = pd.Series(model_rf.feature_importances_).sort_values(ascending=False)
feature_imp

import seaborn as sns
# Creating a bar plot
fig = plt.figure(figsize=(11, 5.24), dpi = 190)
sns.barplot(x=feature_imp.index, y=feature_imp)
# Add labels to your graph
axis_font = {'fontname':'Times New Roman', 'size':'18'}
plt.ylabel('Feature Importance Score', **axis_font)
plt.xlabel('Features', **axis_font)
plt.title("Visualizing Important Features", **axis_font)
plt.legend()
plt.show()

#Make predictions
predictions_rf = model_rf.predict_proba(X)
pp3 = pd.DataFrame(predictions_rf[:,1])
for i in range(9495):
 if pp3[0][i] > 0.5:
    pp3[0][i] = 1
 else:
    pp3[0][i] = 0
tb3 = pd.crosstab(pp3[0], y['tb'])
ac3 = accuracy_score([ 1 if p > 0.5 else 0 for p in  predictions_rf[:,1] ], y)

# AUC
predictions_rf = model_rf.predict_proba(X)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y, predictions_rf[:,1])
auc(false_positive_rate, true_positive_rate)

#-------------------------------Neural Network---------------------------    
model_nn = MLPClassifier()
model_nn.fit(X,y)

model_nn.score(X, y)
model_nn.coefs_
dir(model_nn)

print("Training set score: %f" % model_nn.score(X, y))

#print("Training set score: %f" % model_nn.score(X_train, y_train))
#print("Test set score: %f" % mlp.score(X_test, y_test))

#fig, axes = plt.subplots(2)
fig, axes = plt.figure(figsize=(4.33, 3.75), dpi = 150)
# use global min / max to ensure all weights are shown on the same scale
vmin, vmax = model_nn.coefs_[0].min(), model_nn.coefs_[0].max()
for coef, ax in zip(model_nn.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(41, 1), cmap=plt.cm.gray, vmin=.5 * vmin,
               vmax=.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()

import matplotlib.pyplot as plt

def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    '''
    Draw a neural network cartoon using matplotilb.
    
    :usage:
        >>> fig = plt.figure(figsize=(12, 12))
        >>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2])
    
    :parameters:
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
        - layer_sizes : list of int
            List of layer sizes, including input and output dimensionality
    '''
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in xrange(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in xrange(layer_size_a):
            for o in xrange(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
                ax.add_artist(line)


fig66 = plt.figure(figsize=(12, 12))
ax = fig66.gca()
ax.axis('off')



from draw_neural_net_ import draw_neural_net
draw_neural_net(ax, .1, .9, .1, .9, [2, 2, 1],
model_nn.coefs_, 
model_nn.intercepts_,
model_nn.n_iter_,
model_nn.loss_,
np, plt)
plt.savefig('fig66_nn.png')

#Make predictions
predictions_nn = model_nn.predict_proba(X)
pp4 = pd.DataFrame(predictions_nn[:,1])
for i in range(9495):
 if pp4[0][i] > 0.28:
    pp4[0][i] = 1
 else:
    pp4[0][i] = 0     
tb4 = pd.crosstab(pp4[0], y['tb'])
ac4 = accuracy_score([ 1 if p > 0.28 else 0 for p in  predictions_nn[:,1] ], y['tb'])   

# AUC
predictions_nn = model_nn.predict_proba(X)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y, predictions_nn[:,1])
auc(false_positive_rate, true_positive_rate)


