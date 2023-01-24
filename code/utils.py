import scipy.signal as signal
import pandas as pd
import os
import pickle
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import Counter
import tensorflow.keras.backend as K
from tensorflow import keras
import seaborn as sns
from sklearn.metrics import (confusion_matrix, roc_auc_score)
from multiprocessing import Pool
from itertools import product
from functools import partial
import multiprocessing as mp
from multiprocessing import Process, Value, Array
from itertools import chain 
from tensorflow.keras.utils import Sequence
from sklearn.manifold import TSNE
import seaborn as sns
#from bioinfokit.visuz import cluster
from sklearn.decomposition import PCA
#from distinctipy import distinctipy
from scipy import stats
import time, re, os, io
from itertools import chain
import random
from scipy import stats
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import keras
import keras.utils
path = os.getcwd()
    
def getMin(arr, n):
    '''
    This method returns the minimum element in an array
    '''
    res = arr[0]
    for i in range(1,n):
        res = min(res, arr[i])
    return res



def NormalizeF0(F0, scaler):
    F0 = F0.to_numpy()
    shape1 = F0.shape[0]
    shape2 = F0.shape[1]
    F0 = F0.flatten()
    F0 = np.array(F0).reshape(-1, 1)
    F0 = scaler.transform(F0)
    F0 = IntegerEncodeF0(F0_to_ID, F0)
    F0 = np.reshape(F0, (shape1, shape2))
    return F0

def returnEndTokenIndex(arr):
    
    arr=np.round(np.array(arr))
    if 9.0 in arr:
        itemindex = np.where(arr==9.0)
        return itemindex[0][0]
    else:
        return 0

def SaveResults_XLSX(results, Speaker, ID):
    results = np.array(results)
    with open('Speak_'+Speaker+'_AU'+ID+'.csv', 'w', newline='') as file:
        mywriter = csv.writer(file, delimiter=',')
        mywriter.writerows(results)

def GetSpeakerArray(ID, Speaker_to_ID):
   arr = []
   for i in range(100):
      arr.append(ID)
   arrAll = []
   for i in range(10):
       arrAll.append(arr)
   arr=arrAll
   arrAll=[]
   for i in range(50):
       arrAll.append(arr)
   arr=arrAll
   arr=np.array(arr)
   arr = IntegerEncodeAU(Speaker_to_ID, arr)
   return arr


def encode(E1, E2, E3, E4):
    return E1, E2, E3, E4


def tf_encode(x, z1, z2, z4):
    result_x, result_z1, result_z2, result_z4 = tf.py_function(encode, [x, z1, z2, z4],
                                                               [tf.int64, tf.int64, tf.int64, tf.int64])
    result_x.set_shape([None])
    result_z1.set_shape([None])
    result_z2.set_shape([None])
    result_z4.set_shape([None])
    return result_x, result_z1, result_z2, result_z4

def PadF0_1000(F0):
    print('F0 shape: ', F0.shape)
    F0 = F0.tolist()
    to_return = []
    for i in F0:
        to_return.append(i)
    j =  1000 - len(to_return)
    for x in range(j):
        to_return.append(0)
    print('shape of to_return: ', np.array(to_return).shape)
    return np.array(to_return)
    

def TurnIPU_MergedWords(matrix):
    to_return = []
    for i in matrix:
        word = i.flatten()
        word = list(word[~np.isnan(word)])
        word = list(filter((7.0).__ne__, word))
        word = list(filter((9.0).__ne__, word))
        word = list(filter((8.0).__ne__, word))
        word.insert(0, 8.0)
        word.append(9.0)
        if len(word)>101:
            word = word[:100]
        if len(word)<101:
            while len(word)<101:
                word.append(7.0)
        to_return.append(word)
    return np.array(to_return)

def replace_with_dict_2(ar, dic):
    k = np.array(list(dic.keys()))
    v = np.array(list(dic.values()))
    sidx = k.argsort()
    toReturn  = v[sidx[np.searchsorted(k,ar,sorter=sidx)]]
    return toReturn


def IntegerEncodeAU(Vocab, DataFrame):
    integer_encoded = replace_with_dict_2(DataFrame, Vocab)
    return integer_encoded

def IntegerEncodeF0(Vocab, DataFrame):
    integer_encoded = replace_with_dict_2(DataFrame, Vocab)
    return integer_encoded

def IntegerDecode(Vocab, DataFrame):
    integer_encoded = replace_with_dict_2(DataFrame, Vocab)
    return integer_encoded


def OneHot_Encode(integer_encoded, LENvocab):
    onehot_encoded = list()
    for value in integer_encoded:
           encodedWord = []
           for v in value:
               number = [0 for _ in range((LENvocab))]
               if v == 0:
                   encodedWord.append(number)
               elif v != 0:
                   number[(v)] = 1
                   encodedWord.append(number)
           onehot_encoded.append(encodedWord)
    return np.array(onehot_encoded)


def classifyAU(listAU, threshold):
    toreturn = []
    for i in listAU:
        if i <=threshold:
            a=0
            toreturn.append(a)
        else:
            a=1
            toreturn.append(a)
    return toreturn


def convert2DListto1DList(list):
    result = []
    for element in list:
        for subElement in element:
            result.append( subElement )
    return result

def one_hot_decode_sequence(sequence):
    arr = []
    for i in range(100):
        encoded_integer = np.argmax(sequence[0, i, :])
        AU_value = ID_to_AU[encoded_integer]
        arr.append(AU_value)
    return arr


def formatDF_out(DFAllData_AU01_np, modality):
    AU01_out = []
    for count, i in enumerate(DFAllData_AU01_np):
        new = []
        for ii in i:
            if ii != 7.0 :
                new.append(ii)
        if modality == 'target':
            new.append(9.0)
        if len(new)>50:
            new = new[:50]
        else:
            while len(new)<50:
                new.append(7.0)
        AU01_out.append(new)
    return AU01_out
                                                                                                                                                

def loss_func(targets, logits):
    crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    mask = tf.math.logical_not(tf.math.equal(targets, 0))
    mask = tf.cast(mask, dtype=tf.int64)
    loss = crossentropy(targets, logits, sample_weight=mask)
    return loss

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles( np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps
        self.warmup_steps = tf.cast(self.warmup_steps, tf.float32)

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = tf.cast(step,tf.float32) * (self.warmup_steps ** -1.5)

        return tf.cast(tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2), tf.float32)

class LearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Learning rate schedule."""
  def __init__(self, learning_rate, hidden_size, warmup_steps):
    """Constructor.
    Args:
      learning_rate: float scalar, the base learning rate.
      hidden_size: int scalar, the hidden size of continuous representation.
      warmup_steps: int scalar, the num of warm-up steps
    """
    super(LearningRateSchedule, self).__init__()
    self._learning_rate = learning_rate
    self._hidden_size = hidden_size
    self._warmup_steps = tf.cast(warmup_steps, 'float32')

  def __call__(self, global_step):
    """Computes learning rate with linear warmup and rsqrt decay.
    Args:
      global_step: int scalar tensor, the current global step.
    Returns:
      learning_rate: float scalar tensor, the learning rate as a function of
        the input `global_step`.
    """
    global_step = tf.cast(global_step, 'float32')
    learning_rate = self._learning_rate
    learning_rate *= (self._hidden_size**-0.5)
    # linear warmup
    learning_rate *= tf.minimum(1.0, global_step / self._warmup_steps)
    # rsqrt decay
    learning_rate /= tf.sqrt(tf.maximum(global_step, self._warmup_steps))
    return learning_rate
def get_config(self):
    config = {
          'd_model': self.d_model,
          'warmup_steps': self.warmup_steps,          
      }
    return config


def reshape(AU01_in, AU01_target):
        AU01_in = np.array(AU01_in)
        AU01_target = np.array(AU01_target)
        AU01_in = np.reshape(AU01_in, (AU01_in.shape[0], 10, AU01_in[0].shape[1], AU01_in[0].shape[2]))
        AU01_target = np.reshape(AU01_target, (AU01_target.shape[0],10, AU01_target[0].shape[1], AU01_target[0].shape[2]))
        return AU01_in, AU01_target

    
def reshape_WordLevel(AU01_in, AU01_target):
        AU01_in = np.array(AU01_in)
        AU01_target = np.array(AU01_target)
        AU01_in = np.reshape(AU01_in, (AU01_in.shape[0], AU01_in[0].shape[1], AU01_in[0].shape[2]))
        AU01_target = np.reshape(AU01_target, (AU01_target.shape[0], AU01_target[0].shape[1], AU01_target[0].shape[2]))
        return AU01_in, AU01_target

                                        
def OneHot_Encode(integer_encoded, LENvocab):
        onehot_encoded = []
        for value in integer_encoded:
                encodedWord = []
                for v in value:
                        number = [0 for _ in range((LENvocab))]
                        if v == 0:
                                encodedWord.append(number)
                        elif v != 0:
                                number[(v)] = 1
                                encodedWord.append(number)
                onehot_encoded.append(encodedWord)
        return np.array(onehot_encoded)


def cropOutputs(x):
        #x[0] is decoded at the end
        #x[1] is inputs
        #both have the same shape
        #padding = 1 for actual data in inputs, 0 for 0

        padding =  K.cast(K.not_equal(x[1],0), dtype=K.floatx())
        #if you have zeros for non-padded data, they will lose their backpropagation
        print('inside cropOutputs, X : ', x)
        print('inside cropOutputs, padding: ', padding)

        return x[0]*padding
                                        

def nospecial(text):
       import re
       text = re.sub("[^a-zA-Z0-9]+", "",text)
       return text

def load_model(model_filename, model_weights_filename):
       with open(model_filename, 'r', encoding='utf8') as f:
           model = tf.keras.models.model_from_json(f.read())
       model.load_weights(model_weights_filename)
       return model


def load_preprocess(path):	
       with open(path, mode='rb') as in_file:
           return pickle.load(in_file)


def intersection(lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3         


def getIPUNumber(StartEndIPU, StartEndWord, filenameIPU, filenamesF0, i):

        IPUNumber = []
        filename_IPU = filenameIPU[i]
        startIPU = float(StartEndIPU[i][0])
        endIPU = float(StartEndIPU[i][1])

        for j, interval2 in enumerate(StartEndWord):
                        startWord = float(interval2.split()[0])
                        endWord = float(interval2.split()[1])
                        word = interval2.split()[2]
                        filename_word = filenamesF0[j]
                
                        if(filename_IPU == filename_word):
                                if(startIPU<= startWord and endWord<=endIPU):
                                        IPUNumber.append(i)
                                        
        return IPUNumber


    
def removeTokens(array):
    arr = []
    array = [arr.append(i[i<8.]) for i in array]
    arr = [(i.tolist()) for i in arr]
    return arr

def removeTokens_F0(array):
     arr = []
     array = [arr.append(i[i!=-1.0]) for i in array]
     arr = [(i.tolist()) for i in arr]
     return arr
 
def removePadding(array, MODE):
    arr = []
    if MODE != 'F0':
        array = [arr.append(i[i!=7.0]) for i in array]
    else:
        array = [arr.append(i[i!=-1.0]) for i in array]
    #arr = [(i.tolist()) for i in arr]
    return arr

                            
             
def removePAD0(array):
     arr = []
     array = [arr.append(i[i!=0]) for i in array]
     arr = [(i.tolist()) for i in arr]
     return arr
 
def Min_Max_Arr(array):
    array = list(chain.from_iterable(array))
    return min(array), max(array)

def make2DLists_Equal(lst):
    # Calculate length of maximal list
    n = len(max(lst, key=len))
    # Make the lists equal in length
    lst_2 = [x + [None]*(n-len(x)) for x in lst]
    # Convert it to a NumPy array
    a = np.array(lst_2)
    return a


def CountShapes(_2DArray):
    count = []
    for i in _2DArray:
        count.append(len(i))
    return count

def reshape2D(_1DArray, _2Dshapes):
    _2DArray = []
    index = 0
    for i in _2Dshapes:
        _1DArr = []
        for j in range(0, i):
            _1DArr.append(_1DArray[index])
            index += 1
        _2DArray.append(_1DArr)
    _2DArray = np.array(_2DArray)
    return _2DArray


def addTokens(_2DArray, MODE):
    for i in _2DArray:
        if MODE != 'F0':
            i.insert(0, 8)
            i.append(9)
    _2DArray = np.array([np.array(ii) for ii in _2DArray])
    return _2DArray

def addPadding(_2DArray, MODE):
    for i in _2DArray:
        while len(i) <100:
            if MODE == 'F0':
                i.append(-1.0)
            else:
                i.append(7.0)
    return _2DArray


def preprocess(trans, scaler, AU__, j, VOCAB, stepsize, num_dec, AU_name):
    real_translation = []
    for w in AU__[j][1:]:
        if w == num_dec + 1:
            break
        real_translation.append(w)

    raw = real_translation
    raw = IntegerDecode(VOCAB, raw)
    raw = removePadding(raw, AU_name)
    raw = [(i.tolist()) for i in raw if len(i) != 0]
    raw = np.array(raw).flatten()
    raw = np.array(raw).reshape(-1, 1)
    raw = scaler.inverse_transform(raw).reshape(1, -1)[0]
    raw = raw.flatten()
    raw = np.around((raw - (stepsize / 2)), decimals=1)
    raw = np.abs(raw)

    pred = np.array(trans[1:-1])
    pred = np.ravel(pred)
    pred = IntegerDecode(VOCAB, pred)
    pred = removePadding(pred, AU_name)
    pred = [(i.tolist()) for i in pred if len(i) != 0]
    pred = np.array(pred).flatten()
    pred = np.array(pred).reshape(-1, 1)
    pred = scaler.inverse_transform(pred).reshape(1, -1)[0]
    pred = pred.flatten()
    pred = np.around((pred - (stepsize / 2)), decimals=1)
    pred = np.abs(pred)
    EOS = np.where(pred == 9.)
    if len(EOS) != 0:
        if len(EOS[0]) > 0:
            pred = pred[:EOS[0][0] + 1]

    print('Ground Truth - ', AU_name, ' : ', raw)
    print('Prediction - ', AU_name, ' : ', pred)
    print(' ')
    return pred, raw