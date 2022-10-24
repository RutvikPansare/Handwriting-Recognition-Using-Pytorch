#!/usr/bin/env python
# coding: utf-8

# In[1]:
from sklearn.model_selection import GridSearchCV
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.svm import SVC
import pickle
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import time


# Import Dataset

# In[20]:


import cv2
data = []
labels = []
Devnagari = {"0":0,"1":10,"2":2,"3":3,"4":11,"5":12,"6":13,"7":14,"8":15,"9":9}
Western_Arabic = {"0":16,"1":1,"2":17,"3":18,"4":19,"5":20,"6":21,"7":22,"8":23,"9":9}
f = open("./Data/Subsets/DAT_Representation/dataset_final.dat", "rb+")
dataset = pickle.load(f)
for i in range(len(dataset[0])):
    image = dataset[0][i]
    number = np.argmax(dataset[1][i])
    data.append(image)
    labels.append(number)


# In[3]:


X_train,X_test,y_train,y_test = train_test_split(data,labels, test_size = 0.4, random_state = 1)
X_train,X_test,y_train,y_test = train_test_split(X_train,y_train,test_size = 0.7, random_state = 1)


# In[4]:


X_train = np.array(X_train)
X_test = np.array(X_test)
n_samples,nx,ny = X_train.shape
n_samples_test,nx_test,ny_test = X_test.shape
X_train = X_train.reshape((n_samples,nx*ny))
X_test_ = X_test.reshape((n_samples_test,nx_test*ny_test))


# Train Model

# In[5]:


# svm = LinearSVC()
# svm.fit(X_train,y_train)
# acc = svm.score(X_train,y_train)
# print("Acc: ",acc)


# # In[6]:
#
#
# y_pred_Linear = svm.predict(X_test_)
#
#
# # In[7]:
#
#
# confusion = metrics.confusion_matrix(y_true = y_test, y_pred = y_pred_Linear)
# confusion
#
#
# # In[8]:
#
#
# metrics.accuracy_score(y_true=y_test, y_pred=y_pred_Linear)
#
#
# # In[10]:
#
#
# report = classification_report(y_test, y_pred_Linear)
# print(report)
#
#
# # In[ ]:
#
#
# # K fold Cross validation
# from sklearn.model_selection import cross_val_score
# scores = cross_val_score(svm, X_train,y_train,scoring = 'r2', cv = 5)
# scores
#
#
# # Non-Linear SVM
#
# # In[11]:
#
#
# from sklearn import svm
# svm_rbf = svm.SVC(kernel='rbf')
# svm_rbf.fit(X_train, y_train)
#
#
# # In[12]:
#
#
# y_pred_Non_Linear = svm_rbf.predict(X_test_)
#
#
# # In[13]:
#
#
# print(metrics.accuracy_score(y_true=y_test, y_pred=y_pred_Non_Linear))
#
#
# # In[15]:
#
#
# report = classification_report(y_test, y_pred_Non_Linear)
# print(report)
#
#
# # In[ ]:
#
#
# confusion = metrics.confusion_matrix(y_true = y_test, y_pred = y_pred_Non_Linear)
# confusion
#
#
# # In[ ]:
#
#
# report = classification_report(y_test, y_pred_Non_Linear)
# print(report)


# Grid Search

# In[9]:


model_params = {
    'svm': {
        'model': svm.SVC(),
        'params' : {
            'C': [1,10],
            'kernel': ['rbf','linear'],
            'gamma': [0.3, 0.5,0.1]
        }  
    }
}


# In[ ]:

print('Starting Now...')
start = round(time.time() * 1000,3)
print('Current Time:', start)
scores = []
X_train = X_train[:10000]
y_train = y_train[0:10000]
for model_name, mp in model_params.items():
    clfgrid =  GridSearchCV(mp['model'], mp['params'], cv=5,n_jobs = -1 ,return_train_score=False)
    clfgrid.fit(X_train, y_train)
    scores.append({
        'model': model_name,
        'best_score': clfgrid.best_score_,
        'best_params': clfgrid.best_params_
    })
    
df_grid = pd.DataFrame(scores,columns=['model','best_score','best_params'])
# df_grid
end = round(time.time() * 1000,3)
print('Total time:', end-start)


# Training using best parameters

# In[15]:


model = SVC(C=10, gamma=0.3, kernel="rbf")

model.fit(X_train, y_train)
y_pred = model.predict(X_test_)

# metrics
print("accuracy", metrics.accuracy_score(y_test, y_pred), "\n")
print(metrics.confusion_matrix(y_test, y_pred), "\n")
print('stuff:\n', df_grid)


# In[14]:

#
# print(metrics.accuracy_score(y_true=y_test, y_pred=y_pred))
#
#
# # In[ ]:
#
#
# report = classification_report(y_test, y_pred)
# print(report)
#
#
# # Finding ROC and AUC curves
#
# # In[ ]:
#
#
# from sklearn.calibration import CalibratedClassifierCV
# svm = LinearSVC()
# clf = CalibratedClassifierCV(svm)
# clf.fit(X_train, y_train)
# y_proba_Linear = clf.predict_proba(X_test_)
#
#
# # In[ ]:
#
#
# from sklearn.metrics import roc_curve, auc,roc_auc_score
# roc_auc_score(y_test, y_proba_Linear,multi_class = "ovr")
#
#
# # In[ ]:
#
#
# from sklearn import svm
# from sklearn.multiclass import OneVsRestClassifier
# svm_rbf = OneVsRestClassifier(svm.SVC(kernel='rbf',probability = True))
# svm_rbf.fit(X_train, y_train)
#
#
# # In[ ]:
#
#
# y_pred =svm_rbf.predict(X_test_)
# y_proba_Linear = svm_rbf.predict_proba(X_test_)
#
#
# # In[ ]:
#
#
# classes=np.unique(y_test)
#
#
# # In[ ]:
#
#
# from sklearn.preprocessing import label_binarize
# #binarize the y_values
#
# y_test_binarized=label_binarize(y_test,classes=np.unique(y_test))
#
# # roc curve for classes
# fpr = {}
# tpr = {}
# thresh ={}
# roc_auc = dict()
#
# n_class = 3
#
# for i in range(n_class):
#     fpr[i], tpr[i], thresh[i] = roc_curve(y_test_binarized[:,i], y_proba_Linear[:,i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
#
#     # plotting
#     plt.plot(fpr[i], tpr[i], linestyle='--',
#              label='%s vs Rest (AUC=%0.2f)'%(classes[i],roc_auc[i]))
#
# plt.plot([0,1],[0,1],'b--')
# plt.xlim([0,1])
# plt.ylim([0,1.05])
# plt.title('Multiclass ROC curve')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive rate')
# plt.legend(loc='lower right')
# plt.show()
#
#
# # In[ ]:




