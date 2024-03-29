# -*- coding: utf-8 -*-
import argparse

import collections

import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import chainer
from chainer import serializers, cuda, Variable, optimizers, Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

import sys
import time
import random
import math as mt
import skimage.io as sk
import numpy as np

#from sklearn.manifold import TSNE

os.environ['QT_QPA_PLATFORM']='offscreen'
plt.style.use('ggplot')

p = argparse.ArgumentParser()
p.add_argument('--input', '-i', default='dataset/cross_validation/fold2')
p.add_argument('--model', '-m', default='best_model.npz')
p.add_argument('--gpu', '-g', type=int, default=-1)
p.add_argument('--label', '-y', type=int, default=-1)
p.add_argument('--layer', '-l', default='conv3')
args = p.parse_args()

os.makedirs('results', exist_ok=True)
imsize = 200

# Load images
def loadImages(path):
    imagePathes = list(map(lambda a:os.path.join(path,a),os.listdir(path)))
    images = np.array(list(map(lambda x: sk.imread(x).reshape(imsize, imsize), imagePathes)))
    return(images)

images = {}
images["diff_test"] = loadImages(os.path.join(args.input, "diff/test"))
images["ndiff_test"] = loadImages(os.path.join(args.input, "ndiff/test"))

images["diff_test"] = images["diff_test"].astype('float32')
images["ndiff_test"] = images["ndiff_test"].astype('float32')
images["diff_raw"] = loadImages(os.path.join(args.input, "diff/test"))
images["ndiff_raw"] = loadImages(os.path.join(args.input, "ndiff/test"))


# Pre-Processing
for i in range(len(images["diff_test"])):
    p50 = np.percentile(images["diff_test"][i], 50)
    images["diff_test"][i] /= p50

for i in range(len(images["ndiff_test"])):
    p50 = np.percentile(images["ndiff_test"][i], 50)
    images["ndiff_test"][i] /= p50

images["all"] = np.vstack([images["diff_test"], images["ndiff_test"]]).astype(np.float32)

diff_testnumber = len(images['ndiff_test'])
ndiff_testnumber = len(images['ndiff_test'])
all_number = diff_testnumber + ndiff_testnumber

images["diff_test"] = images['diff_test'].reshape(diff_testnumber, 1, imsize, imsize)
images["ndiff_test"] = images['ndiff_test'].reshape(ndiff_testnumber, 1, imsize, imsize)
images["diff_raw"] = images['diff_raw'].reshape(diff_testnumber, imsize, imsize)
images["ndiff_raw"] = images['ndiff_raw'].reshape(ndiff_testnumber, imsize, imsize)

x_test = np.vstack([images["diff_test"], images["ndiff_test"]]).astype(np.float32)
y_test = np.append([0]*len(images['diff_test']), [1]*len(images['ndiff_test'])).astype(np.int32)
src_test = np.vstack([images["diff_raw"], images["ndiff_raw"]]).astype(np.float32)

N_test = diff_testnumber + ndiff_testnumber

print('Test Data : ' + str(N_test))

k1_s = 4
f1_s = 155
p1_s = 3
k2_s = 9
f2_s = 130
p2_s = 5
k3_s = 3
f3_s = 152
p3_s = 3
f_units = 2863
dropout = 0.799

def max_pooling1(x):
    return F.max_pooling_2d(x, ksize=p1_s, stride=p1_s)

def max_pooling2(x):
    return F.max_pooling_2d(x, ksize=p2_s, stride=p2_s)

def max_pooling3(x):
    return F.max_pooling_2d(x, ksize=p3_s, stride=p3_s)

def Dropout(x):
    return F.dropout(x, ratio=dropout)

class diff_nn(Chain):
    def __init__(self, k1_s=k1_s, f1_s=f1_s, k2_s=k2_s, f2_s=f2_s, k3_s=k3_s, f3_s=f3_s, f_units=f_units):
        self.k1_s = k1_s
        self.f1_s = f1_s
        self.k2_s = k2_s
        self.f2_s = f2_s
        self.k3_s = k3_s
        self.f3_s = f3_s
        self.f_units = f_units
        self.initializer = chainer.initializers.HeNormal()
        self.size = imsize
        
        super(diff_nn, self).__init__(
                                      conv1=L.Convolution2D(1, self.f1_s, self.k1_s, initialW=self.initializer, initial_bias=None),
                                      conv2=L.Convolution2D(self.f1_s, self.f2_s, self.k2_s, initialW=self.initializer, initial_bias=None),
                                      conv3=L.Convolution2D(self.f2_s, self.f3_s, self.k3_s, initialW=self.initializer, initial_bias=None),
                                      bn1=L.BatchNormalization(self.f1_s),
                                      bn2=L.BatchNormalization(self.f2_s),
                                      bn3=L.BatchNormalization(self.f3_s),
                                      fc1 = L.Linear(None, self.f_units),
                                      fc2 = L.Linear(self.f_units, 2),
                                      )
        self.functions = collections.OrderedDict([
                                                  ('conv1', [self.conv1, self.bn1, F.relu]),
                                                  ('pool1', [max_pooling1]),
                                                  ('conv2', [self.conv2, self.bn2, F.relu]),
                                                  ('pool2', [max_pooling2]),
                                                  ('conv3', [self.conv3, self.bn3, F.relu]),
                                                  ('pool3', [max_pooling3]),
                                                  ('fc1', [self.fc1, F.relu, Dropout]),
                                                  ('fc2', [self.fc2]),
                                                  ('prob', [F.softmax]),
                                                  ])
    
    def calc(self, x, train):
        h = max_pooling1(F.relu(self.bn1(self.conv1(x))))
        h = max_pooling2(F.relu(self.bn2(self.conv2(h))))
        h = max_pooling3(F.relu(self.bn3(self.conv3(h))))
        h = F.relu(self.fc1(h))
        with chainer.using_config('train', train):
            h = Dropout(h)
        h = self.fc2(h)
        return h

    def __call__(self, x, y, train=True):
        x, y = Variable(x), Variable(y)
        h = self.calc(x, train)
        output = F.softmax(h)
        loss = F.softmax_cross_entropy(h, y)
        acc = F.accuracy(h, y)
        del h
        return loss, acc, output

    def makeAct(self, x, train=False):
        
        acts = collections.OrderedDict()
        h = Variable(x)
        acts['input'] = x
        for key, funcs in self.functions.items():
            for func in funcs:
                if func is Dropout:
                    with chainer.using_config('train', train):
                        h = func(h)
                else:
                    h = func(h)
            acts[key] = h.data
    
        return acts

    
model = diff_nn()
serializers.load_npz(args.model, model)

if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()
    model.to_gpu()


tp, tn, fp, fn = 0, 0, 0, 0

with open('results/test.txt', 'w') as f:
    for i in range(N_test):
        print('Analyzing No.' + str(i) + ' image')
        f.write('Analyzing No.' + str(i) + ' image\n')
        x_batch = x_test[i].transpose(0, 1, 2)[np.newaxis, :, :, :]
        src = src_test[i].transpose(0, 1)[:, :, np.newaxis]
        y_batch = np.array(y_test[i]).reshape(1)

        if args.gpu >= 0:
            x_batch = cuda.to_gpu(x_batch)
            y_batch = cuda.to_gpu(y_batch)
    
        loss, acc, prob = model(x_batch, y_batch, train=False)
        f.write('prob is {}\n'.format(prob))

        if y_batch == 0:
            f.write('True label is Diff.\n')
            if acc.data == 1.0:
                tp += 1
                print('Predicted label is correct.')
                f.write('Predicted label is correct.\n')
            elif acc.data == 0.0:
                fn += 1
                print('Predicted label is wrong.')
                f.write('Predicted label is wrong.\n')
        elif y_batch == 1:
            f.write('True label is NDiff.\n')
            if acc.data == 1.0:
                tn += 1
                print('Predicted label is correct.')            
                f.write('Predicted label is correct.\n')
            elif acc.data == 0.0:
                fp += 1
                print('Predicted label is wrong.')
                f.write('Predicted label is wrong.\n')

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    try:
        fscore = 2 * recall * precision / (recall + precision)
    except:
        fscore = 0.0
    print('test accuracy  : {}'.format(accuracy))
    print('test recall    : {}'.format(recall))
    print('test precision : {}'.format(precision))
    print('test f-measure : {}'.format(fscore))
    f.write('test accuracy  : {}\n'.format(accuracy))
    f.write('test recall    : {}\n'.format(recall))
    f.write('test precision : {}\n'.format(precision))
    f.write('test f-measure : {}\n'.format(fscore))
