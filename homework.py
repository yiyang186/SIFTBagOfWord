import cv2 
import numpy as np
import os
from sklearn.svm import LinearSVC
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler

class SIFTBagofWord(object):

    def __init__(self):
        self.fea_det = cv2.FeatureDetector_create("SIFT")
        self.des_ext = cv2.DescriptorExtractor_create("SIFT")
        self.classes_names = []
        self.clf = LinearSVC()
        self.stdSlr = StandardScaler()
        self.width, self.height = 500, 500
        self.voc = None
        self.k = 150

    def fit(self, train_path):
        print "Fitting train set"
        self.classes_names = os.listdir(train_path)
        image_classes, image_paths = self.load_data(train_path)
        des_list, descriptors = self.get_descriptors(image_paths) 
        self.voc, variance = kmeans(descriptors, self.k, 15)
        train_features = self.get_features(self.voc, image_paths, des_list, True)
        self.clf.fit(train_features, np.array(image_classes))

    def predict(self, test_path):
        print "Predicting test set..."
        image_classes, image_paths = self.load_data(test_path)
        des_list, descriptors = self.get_descriptors(image_paths)
        test_features = self.get_features(self.voc, image_paths, des_list, False)
        predictions =  [self.classes_names[i] for i in self.clf.predict(test_features)]
        targets = [self.classes_names[i] for i in image_classes]
        return targets, predictions

    def load_data(self, path):
        print "Loading data ", path
        image_paths = []
        image_classes = []
        names = os.listdir(path)
        label_num = 0
        for name in names:
            _dir = os.path.join(path, name)
            class_path = [os.path.join(_dir, f) for f in os.listdir(_dir)]
            image_paths += class_path
            image_classes += [label_num] * len(class_path)
            label_num += 1
        return image_classes, image_paths

    def get_descriptors(self, image_paths):
        print "Geting descriptors..."
        des_list = []
        for image_path in image_paths:
            img = cv2.imread(image_path)
            img = cv2.resize(img,(self.width, self.height))
            kpts = self.fea_det.detect(img)
            kpts, des = self.des_ext.compute(img, kpts)
            des_list.append((image_path, des))
        descriptors = des_list[0][1]
        for image_path, descriptor in des_list[0:]:
            descriptors = np.vstack((descriptors, descriptor))
        return des_list, descriptors

    def get_features(self, voc, image_paths, des_list, isTrain):
        print "Get features of", "train" if isTrain else "test", "set"
        tf = np.zeros((len(image_paths), self.k), "float64")
        for i in xrange(len(image_paths)):
            words, distance = vq(des_list[i][1], voc)
            for w in words:
                tf[i][w] += 1

        occurences = np.sum((tf > 0) * 1, axis = 0)
        idf = np.array(np.log((1.0 * len(image_paths) + 1) / 
            (1.0 * occurences + 1)), 'float64')
        tfidf = tf * idf

        if isTrain:
            self.stdSlr.fit(tfidf)
        features = self.stdSlr.transform(tfidf)
        return features

if __name__ =='__main__':
    clf = SIFTBagofWord()
    clf.fit('train/')
    targets, predictions = clf.predict('test/')
    print "Result..."
    print "%15s\t%15s" % ("Target", "Predictions")
    for i in xrange(len(predictions)):
        print "%15s\t%15s" % (targets[i], predictions[i])