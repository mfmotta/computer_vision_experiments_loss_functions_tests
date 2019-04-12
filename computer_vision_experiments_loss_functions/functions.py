#adapted from https://github.com/keras-team/keras/blob/master/examples/mnist_siamese.py

import numpy as np
import random
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras import backend as K
import json


num_classes = 10


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def rad_distance(vec):
    x=vec
    return K.sqrt(K.maximum(K.square(x), K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def create_pairs(x, y, digit_indices, shuffle=True): #shape:(n, 2, 28, 28), 2 stands for the pair
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    orig_labels=[]
    
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            orig_labels += [[y[z1], y[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            orig_labels += [[y[z1], y[z2]]]
            labels += [1, 0]
    if shuffle==True:
        
        ind = np.random.choice(len(pairs),len(pairs),replace=False)
        pairs = [pairs[i] for i in ind]
        labels = [labels[i] for i in ind]
        orig_labels = [orig_labels[i] for i in ind]

    return np.array(pairs), np.array(labels), np.array(orig_labels)



def combine_triplet_indices(x): #input x, outputs tuple of indices corresponding to triplets
    start_time = time.time()

    #print(len(list(itertools.permutations(range(len(x)), 3))))
    apns = []
    for elem in list(itertools.permutations(range(len(x)), 3)):
        sap = sorted(elem[:2])
        ng = [elem[2]]
        apns.append(str(sap+ng))
    #print("--- %s seconds ---" % (time.time() - start_time))
    
    return [json.loads(elem) for elem in set(apns)] # faster than list(set(apns)) 


def create_triplets(x, y): #naive implementation used to feed the nn
#7 secs to create ~ 90.000 triplets from a batch of 128 images # selects valid triplets
    start_time = time.time()

    anc = []
    pos = []
    neg = []
    tar = []
    ind_class = {} #dictionary with keys = triplet indices (unique), values= classes
    for indices in combine_triplet_indices(x): 
        ind_a, ind_p, ind_n = indices[0], indices[1], indices[2]
        cla_a, cla_p, cla_n = y[ind_a], y[ind_p], y[ind_n]
        
        if cla_a == cla_p != cla_n:
            #print('classes:', cla_a, cla_p, cla_n)
            #print('indices:', ind_a, ind_p, ind_n)
            #print()
            ind_class[(ind_a, ind_p, ind_n)] = (cla_a, cla_p, cla_n) 
            
            anc.append(x[ind_a])
            pos.append(x[ind_p])
            neg.append(x[ind_n]) 
            tar.append([1])
    #print("--- %s seconds ---" % (time.time() - start_time))    
    return np.array(anc), np.array(pos), np.array(neg),np.array(tar), np.array(tar) #, ind_class



def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    l1 = Flatten()(input)
    l2 = Dense(128, activation='relu')(l1)
    d2 = Dropout(0.1)(l2)
    l3 = Dense(128, activation='relu')(d2)
    d3 = Dropout(0.1)(l3)
    l4 = Dense(128, activation='relu')(d3)
    #l5 = Dense(10, activation='relu')(l4)
    return Model(input, l4) 


def contrastive_loss(y_true, y_pred): #ytrue=label, ypred=distance
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def potential_loss(y_true, y_pred): #shapes if y_true and y_pred must be the same
    
    return K.mean((1-y_true)*K.exp(-10*(1-y_true)*y_pred)/K.maximum(K.epsilon(),y_pred) + 
                  y_true * K.pow(y_pred,3))

def triplet_loss(y_true, y_pred):
    #y_pred: (dp,dn)
    #y_true: (1,1)
    #print(y_pred.shape)
    dp = y_pred[0]
    dn = y_pred[1]
    return K.mean(K.maximum(K.square(dp) - K.square(dn) + 0.85, 0))


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    #print(y_true.shape)
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))



