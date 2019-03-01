#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###
#########################################################

def get_model_train(features_train, labels_train):
    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    return clf

def predict (clf, features_test):
    return clf.predict(features_test)

def get_accuracy(clf, features_test, labels_test):
    return clf.score(features_test, labels_test)

if __name__ == "__main__":
    features_train, features_test, labels_train, labels_test = preprocess()
    
    t0 = time()
    clf = get_model_train(features_train, labels_train)
    print "Training time: ", round(time()-t0, 3), "s"
    t0 = time()
    print "Prediction: ", predict(clf, features_test)
    print "Prediction time: ",  round(time()-t0, 3), "s"
    print "Accuracy: ", round(get_accuracy(clf, features_test, labels_test), 3)
