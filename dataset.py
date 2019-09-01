''' 用于加载数据集 '''

import _pickle as cPickle
import gzip
import numpy as np

class MNIST(object):
    def __init__(self, path):
        '''
        用于加载 MNIST 数据集

        参数：
        path 用于指定 mnist 数据集所在的文件夹路径
        '''
        self.path = path
        self.train_data, self.test_data = self.load(self.path)
        self.loadDataWrapper()

    
    def load(self, path):
        '''
        从文件中读取内容
        '''
        fileObj = open(self.path+'train-images.idx3-ubyte')
        loaded = np.fromfile(file = fileObj, dtype = np.uint8)
        trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)
        fileObj.close()

        fileObj = open(self.path + 'train-labels.idx1-ubyte')
        loaded = np.fromfile(file = fileObj, dtype = np.uint8)
        trY = loaded[8:].reshape((60000)).astype(np.float)
        fileObj.close()
        
        fileObj = open(self.path + 't10k-images.idx3-ubyte')
        loaded = np.fromfile(file = fileObj, dtype = np.uint8)
        teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)
        fileObj.close()

        fileObj = open(self.path + 't10k-labels.idx1-ubyte')
        loaded = np.fromfile(file = fileObj, dtype = np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.float)
        fileObj.close()

        trY = np.asarray(trY)
        teY = np.asarray(teY)
        
        return (trX, trY), (teX, teY)

    def loadDataWrapper(self):
        '''
        将读取的内容转化为 ndarray 的 list
        '''
        train_imgs = [np.reshape(x, (784, 1)) for x in self.train_data[0]]
        train_labels = [vectorize(x) for x in self.train_data[1]]
        self.train_data = list(zip(train_imgs, train_labels))
        
        # valid_imgs = [np.reshape(x, (784, 1)) for x in self.valid_data[0]]
        # valid_labels = [vectorize(x) for x in self.valid_data[1]]
        # self.valid_data = zip(valid_imgs, valid_labels)

        test_imgs = [np.reshape(x, (784, 1)) for x in self.test_data[0]]
        test_labels = [vectorize(x) for x in self.test_data[1]]
        self.test_data = list(zip(test_imgs, test_labels))

def vectorize(x):
    v = np.zeros((10, 1))
    v[int(x)] = 1.0
    return v