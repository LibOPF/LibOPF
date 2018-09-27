import time

import libopf_py
import numpy
from scipy.spatial import distance
from sklearn import datasets
from sklearn.metrics import classification_report, confusion_matrix

digits = datasets.load_digits()

# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))


def run(split):
    n_split = int(split * n_samples)

    print("")
    print("=" * 100)
    print("")

    print("Split: %3.2f" % split)
    print("Size: %d, Classifying Size: %d, Testing Size: %d" % (n_samples, n_split, n_samples - n_split))

    rand = numpy.random.permutation(n_samples)

    random_data = data[rand]
    random_label = digits.target[rand]

    dist = distance.squareform(distance.pdist(random_data, 'sqeuclidean'))
    dist_train, dist_test = numpy.ascontiguousarray(dist[:n_split, :n_split]), numpy.ascontiguousarray(
        dist[:n_split, n_split:])

    label_train, label_test = random_label[:n_split], random_label[n_split:]

    print("-" * 20, "OPF", "-" * 20)

    def opf():
        # OPF only supports 32 bits labels at the moment
        label_train_32 = label_train.astype(numpy.int32)
        label_test_32 = label_test.astype(numpy.int32)

        O = libopf_py.OPF()

        t = time.time()
        O.fit(dist_train, label_train_32, precomputed_distance=True)
        #    O.fit(dist_train, label_train_32, precomputed_distance=True, learning="agglomerative", split=0.8)
        print("OPF: time elapsed in fitting: %f secs" % (time.time() - t))

        t = time.time()
        predicted = O.predict(dist_test)
        print("OPF: time elapsed in predicting: %f secs" % (time.time() - t))

        print("Classification report for OPF:\n%s\n" % (classification_report(label_test_32, predicted)))
        print("Confusion matrix:\n%s" % confusion_matrix(label_test_32, predicted))

    opf()


run(0.8)
