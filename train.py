# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
import numpy as np
import chainer
from chainer import cuda, Variable, optimizers
import chainer.functions as F
import chainer.links as L
from chainer import Link, Chain, ChainList
from chainer import serializers
import sys
import csv
import time
import random
import math as mt
import skimage.io as sk
import argparse
import copy
import collections

os.environ['QT_QPA_PLATFORM']='offscreen'
plt.style.use('ggplot')

starttime = time.time()
ap = argparse.ArgumentParser(description='python train.py')
ap.add_argument('--input', '-i', nargs='?', default='dataset', help='Specify input directory')
ap.add_argument('--gpu', '-g', type=int, default=-1, help='Specify GPU ID (negative value indicates CPU)')
ap.add_argument('--crop_size', '-c', type=int, default=180, help='Specify crop size (int)')
ap.add_argument('--preprocess', '-p', type=int, default=1, help='Specify pre-process mode; 1. median, 2. normalization')
ap.add_argument('--batchsize', '-b', type=int, default=2, help='Specify mini-batch size')
ap.add_argument('--epoch', '-e', type=int, default=100, help='Specify the number of epoch')
args = ap.parse_args()


batchsize = args.batchsize
epoch_num = args.epoch
imsize = 200
pd_size = int((imsize - args.crop_size) / 2)
draw = True
ans = False
print('[Training property]')
print('batchsize  : ' + str(batchsize))
print('epoch      : ' + str(epoch_num))
print('image size : ' + str(imsize))
print('crop size  : ' + str(args.crop_size))
print('pad size   : ' + str(pd_size))


# Load images
def loadImages(path):
    imagePathes = list(map(lambda a:os.path.join(path,a),os.listdir(path)))
    #images = np.array(map(lambda x: cv2.imread(x, 0).reshape(1, 200, 200), imagePathes))
    images = np.array(list(map(lambda x: sk.imread(x).reshape(imsize, imsize), imagePathes)))
    return(images)

# Augmentation
def batch_augmentation(
        images,
        crop_size=(180, 180)
    ):
    """ 2d image batches are cropped from array.
    Args:
        images (np.ndarray)         : Input batch of 2d images array
        crop_size ((int, int))      : Crop patch from array
    Returns:
        aug_images (np.ndarray)  : augmentation 2d image array
    """
    bn, _, y_len, x_len = images.shape
    assert y_len >= crop_size[0]
    assert x_len >= crop_size[1]
    aug_images = np.zeros((bn, 1, crop_size[0], crop_size[1]))

    for n in range(bn):
        # get cropping position (image)
        left = random.randint(0, x_len-crop_size[1]-1) if x_len > crop_size[1] else 0
        top = random.randint(0, y_len-crop_size[0]-1) if y_len > crop_size[0] else 0
        right = left + crop_size[0]
        bottom = top + crop_size[1]
        aug_images[n, 0] = images[n, 0, top:bottom, left:right]

        # get rotating position (image)
        aug_flag = random.randint(0, 3)
        aug_images[n, 0] = np.rot90(aug_images[n, 0], k=aug_flag)

    return np.array(aug_images).astype(np.float32)

# Pre-processing
def image_preprocess(
        image,
        mode = 1
    ):
    """ 2d image batches are cropped from array.
    Args:
        images (np.ndarray)         : Input batch of 2d images array
        mode   (int)                : 1. Median devision method
                                      2. Normalization method
    Returns:
        proc_image (np.ndarray)  : augmentation 2d image array
    """
    if mode == 1:
        image = image / np.percentile(image, 50)
    elif mode == 2:
        mean = image.mean()
        std = image.std()
        image = (image - mean) / std
    else:
        raise ValueError('mode is int value {1, 2}')
    return copy.deepcopy(image)



images = {}
images["diff_train"] = loadImages(args.input + "/diff/train")
images["ndiff_train"] = loadImages(args.input + "/ndiff/train")
images["diff_test"] = loadImages(args.input + "/diff/test")
images["ndiff_test"] = loadImages(args.input + "/ndiff/test")

images["diff_train"] = images["diff_train"].astype('float32')
images["ndiff_train"] = images["ndiff_train"].astype('float32')
images["diff_test"] = images["diff_test"].astype('float32')
images["ndiff_test"] = images["ndiff_test"].astype('float32')

for i in range(len(images["diff_train"])):
    images["diff_train"][i] = image_preprocess(images["diff_train"][i], mode=args.preprocess)
for i in range(len(images["ndiff_train"])):
    images["ndiff_train"][i] = image_preprocess(images["ndiff_train"][i], mode=args.preprocess)
for i in range(len(images["diff_test"])):
    images["diff_test"][i] = image_preprocess(images["diff_test"][i], mode=args.preprocess)
for i in range(len(images["ndiff_test"])):
    images["ndiff_test"][i] = image_preprocess(images["ndiff_test"][i], mode=args.preprocess)


images["all"] = np.vstack([images["diff_train"], images["ndiff_train"], images["diff_test"], images["ndiff_test"]]).astype(np.float32)

diff_trainnumber = len(images['diff_train'])
ndiff_trainnumber = len(images['ndiff_train'])
diff_testnumber = len(images['ndiff_test'])
ndiff_testnumber = len(images['ndiff_test'])
all_number = diff_trainnumber + ndiff_trainnumber + diff_testnumber + ndiff_testnumber

images["diff_train"] = images['diff_train'].reshape(diff_trainnumber, 1, imsize, imsize)
images["ndiff_train"] = images['ndiff_train'].reshape(ndiff_trainnumber, 1, imsize, imsize)
images["diff_test"] = images['diff_test'].reshape(diff_testnumber, 1, imsize, imsize)
images["ndiff_test"] = images['ndiff_test'].reshape(ndiff_testnumber, 1, imsize, imsize)

x_train = np.vstack([images["diff_train"], images["ndiff_train"]]).astype(np.float32)
x_test = np.vstack([images["diff_test"], images["ndiff_test"]]).astype(np.float32)
y_train = np.append([0]*len(images['diff_train']), [1]*len(images['ndiff_train'])).astype(np.int32)
y_test = np.append([0]*len(images['diff_test']), [1]*len(images['ndiff_test'])).astype(np.int32)

N_train = diff_trainnumber + ndiff_trainnumber
N_test = diff_testnumber + ndiff_testnumber


print('[Dataset property]')
print('Data Sets     : ' + str(all_number))
print('Learning Data : ' + str(N_train))
print('Test Data     : ' + str(N_test))



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
        return loss, acc

    
# Optimizer
model = diff_nn()
optimizer = optimizers.Adam()
optimizer.setup(model)

if args.gpu >= 0:
    cuda.get_device(0).use()
    model.to_gpu()

train_loss = []
train_acc  = []
test_loss = []
test_acc  = []

sum_train_error = []
sum_test_error = []
sum_train_accuracy = []
sum_test_accuracy = []

diffErrorList = []
ndiffErrorList = []

conv1_W = []
conv2_W = []
conv3_W = []
l4_W = []
l5_W = []

best_accuracy = 0

# Learning loop
for epoch in range(1, epoch_num+1):
    print('---------------------------')
    print('epoch', epoch)

    # training
    perm = np.random.permutation(N_train)
    train_sum_accuracy = 0
    train_sum_loss = 0
    for i in range(0, N_train, batchsize):
        x_batch = x_train[perm[i:i+batchsize]]
        x_batch = batch_augmentation(x_batch, crop_size=(args.crop_size, args.crop_size))
        y_batch = y_train[perm[i:i+batchsize]]
        real_batchsize = len(x_batch)

        if args.gpu >= 0:
            x_batch = cuda.to_gpu(x_batch)
            y_batch = cuda.to_gpu(y_batch)

        loss, acc = model(x_batch, y_batch, train=True)
        optimizer.target.cleargrads()
        loss.backward()
        optimizer.update()

        train_loss.append(loss.data)
        train_acc.append(acc.data)

        train_sum_loss += float(cuda.to_cpu(loss.data)) * real_batchsize
        train_sum_accuracy += float(cuda.to_cpu(acc.data)) * real_batchsize


    # sum_train_error.append(1 - (train_sum_accuracy / N_train))
    sum_train_error.append(train_sum_loss / N_train)
    sum_train_accuracy.append(train_sum_accuracy / N_train)

    # evaluation
    test_sum_accuracy = 0
    test_sum_loss     = 0
    diff_error = 0
    ndiff_error = 0

    for i in range(0, N_test):
        x_batch = x_test[i].reshape(1, 1, imsize, imsize)
        if pd_size != 0:
            x_batch = x_batch[0, 0, pd_size:-pd_size, pd_size:-pd_size].reshape(1, 1, args.crop_size, args.crop_size)
        y_batch = np.array(y_test[i]).reshape(1)
        real_batchsize = len(x_batch)

        if args.gpu >= 0:
            x_batch = cuda.to_gpu(x_batch)
            y_batch = cuda.to_gpu(y_batch)

        loss, acc = model(x_batch, y_batch, train=False)

        test_loss.append(loss.data)
        test_acc.append(acc.data)
        test_sum_loss     += float(cuda.to_cpu(loss.data)) * real_batchsize
        test_sum_accuracy += float(cuda.to_cpu(acc.data)) * real_batchsize

    print('---------------------------')
    print('train mean loss={}, accuracy={}'.format(train_sum_loss / N_train, train_sum_accuracy / N_train))
    print('test  mean loss={}, accuracy={}'.format(test_sum_loss / N_test, test_sum_accuracy / N_test))

    if test_sum_accuracy / N_test >= best_accuracy:
        best_epoch = epoch
        best_accuracy = test_sum_accuracy / N_test
        serializers.save_npz("best_model_{}.npz".format(args.input[-5:]), model)

    sum_test_error.append(test_sum_loss / N_test)
    sum_test_accuracy.append(test_sum_accuracy / N_test)

print('best_epoch:' + str(best_epoch))
print('best_accuracy:' + str(best_accuracy))




serializers.load_npz("best_model_{}.npz".format(args.input[-5:]), model)

# TP, TN
TP = 0
TN = 0
FP = 0
FN = 0
for i in range(0, N_test):
    x_batch = x_test[i].reshape(1, 1, imsize, imsize)
    if pd_size != 0:
        x_batch = x_batch[0, 0, pd_size:-pd_size, pd_size:-pd_size].reshape(1, 1, args.crop_size, args.crop_size)
    y_batch = np.array(y_test[i]).reshape(1)
    real_batchsize = len(x_batch)

    if args.gpu >= 0:
        x_batch = cuda.to_gpu(x_batch)
        y_batch = cuda.to_gpu(y_batch)

    loss, acc = model(x_batch, y_batch, train=False)

    if acc.data == 1:
        if y_batch == 0:
            TP += 1
        elif y_batch == 1:
            TN += 1
    elif acc.data == 0:
        if y_batch == 0:
            FN += 1
        elif y_batch == 1:
            FP += 1

print('TP:' + str(TP))
print('TN:' + str(TN))
print('FP:' + str(FP))
print('FN:' + str(FN))

plt.style.use('fivethirtyeight')
def draw_digit2(data, n, ans, recog):
    size = imsize
    plt.subplot(10, 10, n)
    Z = data.reshape(size,size)
    Z = Z[::-1,:]
    plt.xlim(0,99)
    plt.ylim(0,99)
    plt.pcolor(Z)
    if ans == 0:
        if recog == 0:
            plt.title("ans=diff, recog=diff", size=8)
        else:
            plt.title("ans=diff, recog=ndiff", size=8)
    else:
        if recog == 0:
            plt.title("ans=ndiff, recog=diff", size=8)
        else:
            plt.title("ans=ndiff, recog=ndiff", size=8)
    plt.gray()
    plt.tick_params(labelbottom="off")
    plt.tick_params(labelleft="off")

if ans:
    plt.figure(figsize=(15,15))
    serializers.load_npz('bestepoch.model', model)
    if args.gpu >= 0:
        model.to_gpu()
    cnt = 0

    for idx in range(N_test):
        x_batch = x_test[idx].reshape(1, 1, imsize, imsize)
        if args.gpu >= 0:
            x_data = cuda.to_gpu(x_batch)
        x = Variable(x_data)
        h1 = F.max_pooling_2d( F.relu( model['layer{}'.format(1)](model['layer{}'.format(0)](x)) ), ksize=3, stride=3 )
        h2 = F.max_pooling_2d( F.relu( model['layer{}'.format(3)](model['layer{}'.format(2)](h1)) ), ksize=3, stride=3 )
        h3 = F.dropout( F.relu(model['layer{}'.format(4)](h2)), ratio=0.1, train=False)
        y = model['layer{}'.format(5)](h3)

        y_batch = cuda.to_cpu(y.data)
        if y_test[idx] != np.argmax(y_batch):
            cnt+=1
            draw_digit2(x_test[idx], cnt, y_test[idx], np.argmax(y_batch))
            figname = 'false_figures.png'
            plt.savefig(figname)


# グラフの描画
if draw:

    # draw accuracy
    plt.figure(figsize=(8,6))
    plt.plot(range(1, epoch_num+1), sum_train_accuracy)
    plt.plot(range(1, epoch_num+1), sum_test_accuracy)
    plt.xlim([1, epoch_num])
    plt.ylim([0, 1])
    plt.legend(["train_acc","test_acc"],loc='upper right')
    plt.title("Accuracy")
    plt.plot()
    figname = 'Accuracy_' + str(epoch_num) + '.pdf'
    plt.savefig(os.path.join('results', figname))

    # draw error rate
    plt.figure(figsize=(8,6))
    plt.plot(range(1, epoch_num+1), sum_train_error)
    plt.plot(range(1, epoch_num+1), sum_test_error)
    plt.legend(["train_error","test_error"],loc='upper right')
    plt.xlim([1, epoch_num])
    plt.ylim([0, 10])
    plt.title("Error Rate")
    plt.plot()
    figname = 'Error_' + str(epoch_num) + '.pdf'
    plt.savefig(os.path.join('results', figname))
