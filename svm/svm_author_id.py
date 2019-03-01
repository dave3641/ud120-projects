#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###

#########################################################

def my_svm(features_train, features_test, labels_train, labels_test, kernel='linear', C=1.0):
    clf = SVC(kernel=kernel, C=C)
    
    t0 = time()
    clf.fit(features_train, labels_train)
    print "\nTraining time: ", round(time()-t0,3), "s"
    
    t0 = time()
    pred = clf.predict(features_test)
    print "\nPredicting time: ", round(time()-t0,3), "s"
    
    accuracy = round(accuracy_score(pred, labels_test), 3)
    
    print '\nAccuracy = ', accuracy
    return pred

#features_train2 = features_train[:len(features_train)/100]
#labels_train2 = labels_train[:len(labels_train)/100]

#for C in [10, 100, 1000, 10000]:
#    print 'C =', C,
pred = my_svm(features_train, features_test, labels_train, labels_test, 'rbf', C=10000)
#    print '\n'

#print pred[10]
#print pred[26]
#print pred[50]

print sum(pred)