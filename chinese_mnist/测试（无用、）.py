
# In[2]:打开数据集

import numpy as np

import scipy.io as scio

import matplotlib as mpl

import matplotlib.pyplot as plt



file_path ="C:/Users/70951/Desktop/mnist-original.mat"

mnist = scio.loadmat(file_path)

mnist.keys()



# In[3]:数据整理

X, y = mnist["data"], mnist["label"]

X = X.transpose()

X.shape

y = y.transpose()

y.shape

y = y.astype(np.uint8)

y = y.reshape(-1)

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]



# In[4]:梯度下降分类——模型训练



from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(loss='hinge', penalty='l1', max_iter=1, tol=1e-3, random_state=405)

sgd_clf.fit(X_train, y_train)



# In[5]:验证

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

from sklearn.metrics import ConfusionMatrixDisplay



cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")# 每一次验证的正确概率输出

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=3) #使用交叉验证输出预测值

#结果：array([0.87685, 0.8673 , 0.8701 ])，超参数max_iter=1000

#结果：array([0.8363, 0.8468, 0.8527])，超参数max_iter=1        欠拟合

#结果：array([0.8725 , 0.87225, 0.8502 ])，超参数max_iter=10

#结果：array([0.8886 , 0.88775, 0.8921 ])，超参数max_iter=10, penalty=1

#结果：array([0.86285, 0.8832 , 0.84545])，超参数max_iter=10, penalty=elasticnet

conf_mx = confusion_matrix(y_train, y_train_pred)

conf_mx_display = ConfusionMatrixDisplay(conf_mx, display_labels='0123456789')

conf_mx_display.plot(values_format='.4g')

plt.title("SGD:Confusion_matrix", fontsize=14)

#预测测试集

y_test_pred = sgd_clf.predict(X_test)

conf_mx = confusion_matrix(y_test, y_test_pred)

cross_val_score(sgd_clf, X_test, y_test, cv=3, scoring="accuracy")

#结果：array([0.83833233, 0.87038704, 0.86408641])，超参数max_iter=1000

#结果array([0.77684463, 0.84008401, 0.8619862 ])，超参数max_iter=1

#结果array([0.82963407, 0.86978698, 0.87458746])，超参数max_iter=10

plt.matshow(conf_mx, cmap=plt.cm.OrRd) 

plt.colorbar()



# In[6]:分类器指标：准确率与召回率

from sklearn.metrics import precision_score, recall_score



precision_score(y_train.ravel(), y_train_pred, average='macro')

recall_score(y_train.ravel(), y_train_pred, average='macro')



precision_score(y_test.ravel(), y_test_pred, average='macro')

recall_score(y_test.ravel(), y_test_pred, average='macro')



# In[7]随机森林模型



from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(n_estimators=100, max_leaf_nodes=180, random_state=405)

forest_clf.fit(X_train, y_train)

cross_val_score(forest_clf, X_train, y_train, cv=3, scoring="accuracy")# 每一次验证的正确概率输出

y_probas_forest = cross_val_predict(forest_clf, X_train, y_train, cv=3, method="predict_proba")

y_train_pred = cross_val_predict(forest_clf, X_train, y_train, cv=3) #使用交叉验证输出预测值



conf_mx = confusion_matrix(y_train, y_train_pred)

conf_mx_display = ConfusionMatrixDisplay(conf_mx, display_labels='0123456789')

conf_mx_display.plot(values_format='.4g')

plt.title("Random Forest:Confusion_matrix", fontsize=14)



#Grid寻优

from sklearn.model_selection import GridSearchCV

param_grid = [{'n_estimators': [10, 50, 100], 'max_leaf_nodes': [20, 60, 100, 140, 180]}]

forest_clf = RandomForestClassifier(random_state=405)

grid_search = GridSearchCV(forest_clf, param_grid, cv=3, verbose=3, scoring='roc_auc_ovo')

grid_search.fit(X_train, y_train.ravel())

grid_search.best_params_

grid_search.best_score_

# =================================寻优结果==================================

# [CV] max_leaf_nodes=20, n_estimators=10 ..............................

# [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.

# [CV] .. max_leaf_nodes=20, n_estimators=10, score=0.965, total=   1.9s

# [CV] max_leaf_nodes=20, n_estimators=10 ..............................

# [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    1.8s remaining:    0.0s

# [CV] .. max_leaf_nodes=20, n_estimators=10, score=0.968, total=   1.5s

# [CV] max_leaf_nodes=20, n_estimators=10 ..............................

# [Parallel(n_jobs=1)]: Done   2 out of   2 | elapsed:    3.4s remaining:    0.0s

# [CV] .. max_leaf_nodes=20, n_estimators=10, score=0.971, total=   1.5s

# [CV] max_leaf_nodes=20, n_estimators=50 ..............................

# [CV] .. max_leaf_nodes=20, n_estimators=50, score=0.978, total=   6.3s

# [CV] max_leaf_nodes=20, n_estimators=50 ..............................

# [CV] .. max_leaf_nodes=20, n_estimators=50, score=0.979, total=   6.4s

# [CV] max_leaf_nodes=20, n_estimators=50 ..............................

# [CV] .. max_leaf_nodes=20, n_estimators=50, score=0.980, total=   6.3s

# [CV] max_leaf_nodes=20, n_estimators=100 .............................

# [CV] . max_leaf_nodes=20, n_estimators=100, score=0.980, total=  12.4s

# [CV] max_leaf_nodes=20, n_estimators=100 .............................

# [CV] . max_leaf_nodes=20, n_estimators=100, score=0.980, total=  12.1s

# [CV] max_leaf_nodes=20, n_estimators=100 .............................

# [CV] . max_leaf_nodes=20, n_estimators=100, score=0.981, total=  12.6s

# [CV] max_leaf_nodes=60, n_estimators=10 ..............................

# [CV] .. max_leaf_nodes=60, n_estimators=10, score=0.983, total=   1.8s

# [CV] max_leaf_nodes=60, n_estimators=10 ..............................

# [CV] .. max_leaf_nodes=60, n_estimators=10, score=0.984, total=   1.8s

# [CV] max_leaf_nodes=60, n_estimators=10 ..............................

# [CV] .. max_leaf_nodes=60, n_estimators=10, score=0.986, total=   1.8s

# [CV] max_leaf_nodes=60, n_estimators=50 ..............................

# [CV] .. max_leaf_nodes=60, n_estimators=50, score=0.990, total=   8.0s

# [CV] max_leaf_nodes=60, n_estimators=50 ..............................

# [CV] .. max_leaf_nodes=60, n_estimators=50, score=0.990, total=   7.9s

# [CV] max_leaf_nodes=60, n_estimators=50 ..............................

# [CV] .. max_leaf_nodes=60, n_estimators=50, score=0.991, total=   7.9s

# [CV] max_leaf_nodes=60, n_estimators=100 .............................

# [CV] . max_leaf_nodes=60, n_estimators=100, score=0.990, total=  15.6s

# [CV] max_leaf_nodes=60, n_estimators=100 .............................

# [CV] . max_leaf_nodes=60, n_estimators=100, score=0.990, total=  15.6s

# [CV] max_leaf_nodes=60, n_estimators=100 .............................

# [CV] . max_leaf_nodes=60, n_estimators=100, score=0.991, total=  15.6s

# [CV] max_leaf_nodes=100, n_estimators=10 .............................

# [CV] . max_leaf_nodes=100, n_estimators=10, score=0.987, total=   2.0s

# [CV] max_leaf_nodes=100, n_estimators=10 .............................

# [CV] . max_leaf_nodes=100, n_estimators=10, score=0.988, total=   2.0s

# [CV] max_leaf_nodes=100, n_estimators=10 .............................

# [CV] . max_leaf_nodes=100, n_estimators=10, score=0.990, total=   2.0s

# [CV] max_leaf_nodes=100, n_estimators=50 .............................

# [CV] . max_leaf_nodes=100, n_estimators=50, score=0.992, total=   8.7s

# [CV] max_leaf_nodes=100, n_estimators=50 .............................

# [CV] . max_leaf_nodes=100, n_estimators=50, score=0.992, total=   8.7s

# [CV] max_leaf_nodes=100, n_estimators=50 .............................

# [CV] . max_leaf_nodes=100, n_estimators=50, score=0.993, total=   8.7s

# [CV] max_leaf_nodes=100, n_estimators=100 ............................

# [CV]  max_leaf_nodes=100, n_estimators=100, score=0.993, total=  17.2s

# [CV] max_leaf_nodes=100, n_estimators=100 ............................

# [CV]  max_leaf_nodes=100, n_estimators=100, score=0.993, total=  17.3s

# [CV] max_leaf_nodes=100, n_estimators=100 ............................

# [CV]  max_leaf_nodes=100, n_estimators=100, score=0.994, total=  17.2s

# [CV] max_leaf_nodes=140, n_estimators=10 .............................

# [CV] . max_leaf_nodes=140, n_estimators=10, score=0.990, total=   2.1s

# [CV] max_leaf_nodes=140, n_estimators=10 .............................

# [CV] . max_leaf_nodes=140, n_estimators=10, score=0.990, total=   2.1s

# [CV] max_leaf_nodes=140, n_estimators=10 .............................

# [CV] . max_leaf_nodes=140, n_estimators=10, score=0.991, total=   2.1s

# [CV] max_leaf_nodes=140, n_estimators=50 .............................

# [CV] . max_leaf_nodes=140, n_estimators=50, score=0.994, total=   9.2s

# [CV] max_leaf_nodes=140, n_estimators=50 .............................

# [CV] . max_leaf_nodes=140, n_estimators=50, score=0.994, total=   9.3s

# [CV] max_leaf_nodes=140, n_estimators=50 .............................

# [CV] . max_leaf_nodes=140, n_estimators=50, score=0.994, total=   9.3s

# [CV] max_leaf_nodes=140, n_estimators=100 ............................

# [CV]  max_leaf_nodes=140, n_estimators=100, score=0.994, total=  18.3s

# [CV] max_leaf_nodes=140, n_estimators=100 ............................

# [CV]  max_leaf_nodes=140, n_estimators=100, score=0.994, total=  18.3s

# [CV] max_leaf_nodes=140, n_estimators=100 ............................

# [CV]  max_leaf_nodes=140, n_estimators=100, score=0.995, total=  18.3s

# [CV] max_leaf_nodes=180, n_estimators=10 .............................

# [CV] . max_leaf_nodes=180, n_estimators=10, score=0.991, total=   2.2s

# [CV] max_leaf_nodes=180, n_estimators=10 .............................

# [CV] . max_leaf_nodes=180, n_estimators=10, score=0.991, total=   2.2s

# [CV] max_leaf_nodes=180, n_estimators=10 .............................

# [CV] . max_leaf_nodes=180, n_estimators=10, score=0.992, total=   2.2s

# [CV] max_leaf_nodes=180, n_estimators=50 .............................

# [CV] . max_leaf_nodes=180, n_estimators=50, score=0.995, total=   9.7s

# [CV] max_leaf_nodes=180, n_estimators=50 .............................

# [CV] . max_leaf_nodes=180, n_estimators=50, score=0.995, total=   9.8s

# [CV] max_leaf_nodes=180, n_estimators=50 .............................

# [CV] . max_leaf_nodes=180, n_estimators=50, score=0.995, total=   9.6s

# [CV] max_leaf_nodes=180, n_estimators=100 ............................

# [CV]  max_leaf_nodes=180, n_estimators=100, score=0.995, total=  19.0s

# [CV] max_leaf_nodes=180, n_estimators=100 ............................

# [CV]  max_leaf_nodes=180, n_estimators=100, score=0.995, total=  19.0s

# [CV] max_leaf_nodes=180, n_estimators=100 ............................

# [CV]  max_leaf_nodes=180, n_estimators=100, score=0.996, total=  19.3s

# [Parallel(n_jobs=1)]: Done  45 out of  45 | elapsed:  6.7min finished

# =============================================================================



#预测测试集

y_test_pred = forest_clf.predict(X_test)

conf_mx = confusion_matrix(y_test, y_test_pred)

cross_val_score(forest_clf, X_test, y_test, cv=3, scoring="accuracy")

#结果： array([0.89142172, 0.91659166, 0.9369937 ])



# In[8] Gaussian Naive Bayes model

from sklearn.naive_bayes import GaussianNB

gnb_clf = GaussianNB()

gnb_clf.fit(X_train, y_train)

cross_val_score(gnb_clf, X_train, y_train, cv=3, scoring="accuracy")# 每一次验证的正确概率输出

#结果： array([0.5592 , 0.56035, 0.55715])

y_gnb_pred = cross_val_predict(gnb_clf, X_train, y_train.ravel(), cv=3)

conf_mx = confusion_matrix(y_train, y_gnb_pred)

conf_mx_display = ConfusionMatrixDisplay(conf_mx, display_labels='0123456789')

conf_mx_display.plot(values_format='.4g')

plt.title("GaussianNB:Confusion_matrix", fontsize=14)





# In[9] Bernoulli Naive Bayes model

from sklearn.naive_bayes import BernoulliNB

bnb_clf = BernoulliNB(alpha=1)   #alpha平滑参数,0表示不平滑

bnb_clf.fit(X_train, y_train)

cross_val_score(bnb_clf, X_train, y_train, cv=3, scoring="accuracy")# 每一次验证的正确概率输出

#Out[34]: array([0.8252 , 0.82605, 0.8389 ])



y_bnb_pred = cross_val_predict(bnb_clf, X_train, y_train.ravel(), cv=3)

conf_mx = confusion_matrix(y_train, y_bnb_pred)

conf_mx_display = ConfusionMatrixDisplay(conf_mx, display_labels='0123456789')

conf_mx_display.plot(values_format='.4g')

plt.title("BernoulliNB:Confusion_matrix", fontsize=14)





# In[9] SVM（支持向量机）

from sklearn.svm import SVC

svm_clf =  SVC(C=1.0, kernel='poly', gamma='auto', max_iter=50, random_state=405)

svm_clf.fit(X_train, y_train)



cross_val_score(svm_clf, X_train, y_train.ravel(), cv=3, scoring="accuracy")# 每一次验证的正确概率输出

#Out[15]:array([0.4259, 0.4145, 0.454 ]),max_iter=10

#Out[59]:array([0.65915, 0.65565, 0.7004 ]),max_iter=30

#Out[15]:array([0.8491 , 0.8412 , 0.85925]),max_iter=50

cross_val_score(svm_clf, X_test, y_test.ravel(), cv=3, scoring="accuracy")# 每一次验证的正确概率输出

#Out[15]:array([0.90071986, 0.91389139, 0.94029403])



y_svm_pred = cross_val_predict(svm_clf, X_train, y_train.ravel(), cv=3)

conf_mx = confusion_matrix(y_train, y_svm_pred)

conf_mx_display = ConfusionMatrixDisplay(conf_mx, display_labels='0123456789')

conf_mx_display.plot(values_format='.4g')

plt.title("svm:Confusion_matrix", fontsize=14)



#寻优,由于耗时太久，遂放弃

param_grid = [{'C':[1,10,100],

               'kernel':["linear", "poly", "rbf", "sigmoid",],

               'gamma':["auto"],

               'max_iter':[5,30,100]}]

grid_search = GridSearchCV(svm_clf, param_grid, cv=3, verbose=3)

grid_search.fit(X_train, y_train.ravel())

grid_search.best_params_

grid_search.best_score_





















