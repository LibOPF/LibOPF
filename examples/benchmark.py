#!/usr/bin/python
import csv
import gc
import numpy as np
import pylab as pl
from time import time
import libopf_py
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn import datasets


def read_dataset():
    X, y = datasets.load_breast_cancer(return_X_y=True)
    benchmark(X, y, len(y))


def benchmark(data, target, n_samples):
    list_n_samples = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    opf_results = np.zeros((len(list_n_samples), 4))
    svm_results = np.zeros((len(list_n_samples), 4))
    bayes_results = np.zeros((len(list_n_samples), 4))
    linear_results = np.zeros((len(list_n_samples), 4))
    sgd_results = np.zeros((len(list_n_samples), 4))
    tree_results = np.zeros((len(list_n_samples), 4))

    for i, size in enumerate(list_n_samples):
        n_split = int(size * n_samples)
        rand = np.random.permutation(n_samples)
        random_data = data[rand]
        random_label = target[rand]
        data_train, data_test = random_data[:n_split], random_data[n_split:]
        label_train, label_test = random_label[:n_split], random_label[n_split:]

        def _opf():
            label_train_32 = label_train.astype(np.int32)
            label_test_32 = label_test.astype(np.int32)
            O = libopf_py.OPF()
            t = time()
            O.fit(data_train, label_train_32)

            opf_results[i, 3] = time() - t
            t = time()
            predicted = O.predict(data_test)
            opf_results[i, 0] = precision_score(label_test_32, predicted, average='binary')
            opf_results[i, 1] = recall_score(label_test_32, predicted, average='binary')
            opf_results[i, 2] = f1_score(label_test_32, predicted, average='binary')
            gc.collect()

        def _svm():
            clf = svm.SVC(C=1000)
            t = time()
            clf.fit(data_train, label_train)
            svm_results[i, 3] = time() - t
            predicted = clf.predict(data_test)
            svm_results[i, 0] = precision_score(label_test, predicted, average='binary')
            svm_results[i, 1] = recall_score(label_test, predicted, average='binary')
            svm_results[i, 2] = f1_score(label_test, predicted, average='binary')
            gc.collect()

        def _bayes():
            clf = GaussianNB()
            t = time()
            clf.fit(data_train, label_train)
            bayes_results[i, 3] = time() - t
            predicted = clf.predict(data_test)
            bayes_results[i, 0] = precision_score(label_test, predicted, average='binary')
            bayes_results[i, 1] = recall_score(label_test, predicted, average='binary')
            bayes_results[i, 2] = f1_score(label_test, predicted, average='binary')
            gc.collect()

        def _linear():
            clf = LogisticRegression(C=1, penalty='l2')
            t = time()
            clf.fit(data_train, label_train)
            linear_results[i, 3] = time() - t
            predicted = clf.predict(data_test)
            linear_results[i, 0] = precision_score(label_test, predicted, average='binary')
            linear_results[i, 1] = recall_score(label_test, predicted, average='binary')
            linear_results[i, 2] = f1_score(label_test, predicted, average='binary')
            gc.collect()

        def _sgd():
            clf = SGDClassifier(loss="hinge", penalty="l2")
            t = time()
            clf.fit(data_train, label_train)
            linear_results[i, 3] = time() - t
            predicted = clf.predict(data_test)
            sgd_results[i, 0] = precision_score(label_test, predicted, average='binary')
            sgd_results[i, 1] = recall_score(label_test, predicted, average='binary')
            sgd_results[i, 2] = f1_score(label_test, predicted, average='binary')
            gc.collect()

        def _tree():
            clf = tree.DecisionTreeClassifier()
            t = time()
            clf.fit(data_train, label_train)
            tree_results[i, 3] = time() - t
            predicted = clf.predict(data_test)
            tree_results[i, 0] = precision_score(label_test, predicted, average='binary')
            tree_results[i, 1] = recall_score(label_test, predicted, average='binary')
            tree_results[i, 2] = f1_score(label_test, predicted, average='binary')
            gc.collect()

        _opf()
        _svm()
        _bayes()
        _linear()
        _sgd()
        _tree()

    pl.figure()
    pl.plot(list_n_samples, opf_results[:, 2], label="OPF")
    pl.plot(list_n_samples, svm_results[:, 2], label="SVM RBF")
    pl.plot(list_n_samples, bayes_results[:, 2], label="Naive Bayes")
    pl.plot(list_n_samples, linear_results[:, 2], label="Logistic Regression")
    pl.plot(list_n_samples, sgd_results[:, 2], label="SGD")
    pl.plot(list_n_samples, tree_results[:, 2], label="Decision Trees")
    pl.legend(loc='lower right', prop=dict(size=8))
    pl.xlabel("Training set size")
    pl.ylabel("F1 score")
    # pl.title("Precision")
    pl.show()


read_dataset()
