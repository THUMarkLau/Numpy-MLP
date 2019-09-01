''' 用于定义和网络相关的函数 '''

import numpy as np 
import random
import dataset as dst
import json
import sys

class CrossEntropyCost(object):
    ''' 交叉熵损失 '''
    @staticmethod
    def fn(a, y):
        ''' 计算交叉熵 '''
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))
    
    @staticmethod
    def delta(z, a, y):
        ''' 返回输出层的误差 '''
        return (a - y)


class QuodraticCost(object):
    ''' 二次和损失 '''
    @staticmethod
    def fn(a, y):
        ''' 计算二次和损失 '''
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z, a, y):
        ''' 返回输出层误差 '''
        return (a - y) * dsigmoid(z)

def sigmoid(x):
    ''' sigmoid激活函数'''
    return 1.0 / (1.0 + np.exp(-x))

def dsigmoid(x):
    ''' sigmoid函数的导数 '''
    return sigmoid(x) * (1 - sigmoid(x))

class Network(object):
    def __init__(self, size, cost_func, init = 'default'):
        '''
        神经网络
        参数：
            size : 定义神经网络的结构
            cost : 定义计算输出结果的损失函数
            init : 初始化权重的方法，'default'为正态分布，'gauss'为高斯分布
        '''
        # 神经网络的层数
        self.layers_num = len(size)
        # 神经网络的结构，即每层的神经元数目
        self.arch = size
        # 初始化权重
        self.initWeight(init)
        # 损失函数
        self.cost = cost_func

    def initWeight(self, dist):
        ''' 初始化权重 '''
        if dist == 'default':
            ''' 按正态分布初始化权重 '''
            self.biases = [np.random.randn(y, 1) for y in self.arch[1:]]
            self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(self.arch[:-1], self.arch[1:])]
        elif dist == 'gauss':
            ''' 按照高斯分布初始化权重，均值为0，标准差为1 '''
            self.biases = [np.random.randn(y, 1) for y in self.arch[1:]]
            self.weights = [np.random.randn(y, x) for x, y in zip(self.arch[:-1], self.arch[1:])]

    def forward(self, x):
        ''' 前向传播 '''
        a = x
        self.outs = []
        self.activations = [x]
        for b, w in zip(self.biases, self.weights):
            out = np.dot(w, a) + b
            self.outs.append(out)
            a = sigmoid(out)
            self.activations.append(a)
        return a
    
    def backprop(self, x, y):
        ''' 反向传播 '''
        # 生成和biases、weights相同大小的0矩阵
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        self.forward(x)

        # 反向传播
        delta = (self.cost).delta(self.outs[-1], self.activations[-1], y)

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, self.activations[-2].transpose())

        for l in range(2, self.layers_num):
            out = self.outs[-l]
            sp = dsigmoid(out)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, self.activations[-l-1].transpose())
        
        return (nabla_b, nabla_w)

    def SGD(self, training_data, epochs, mini_batch_size, lr,
            lmbda = 0.0, evaluation_data = None, monitor_evaluation_cost = False,
            monitor_evaluation_accuracy = False, monitor_training_cost = False,
            monitor_training_accuracy = False):
        # 验证集数量
        if evaluation_data:
            n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []

        for j in range(epochs):
            # 打乱训练数据
            random.shuffle(training_data)
            # 划分mini batch
            mini_batches = [training_data[k: k+mini_batch_size] for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.updateMiniBatch(mini_batch, lr, lmbda, len(training_data))
            print("Epoch%s training complete" % (j))

            if monitor_training_cost:
                cost = self.totalCost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data:{}/{}".format(accuracy, n))
            
            if monitor_evaluation_cost:
                cost = self.totalCost(evaluation_data, lmdba, convert=True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data:{}".format(cost))
            
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data,convert = True)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {}/{}".format(accuracy, n_data))
            self.save('./nn_2.json')
        return (evaluation_cost, evaluation_accuracy, training_cost, training_accuracy)

    def updateMiniBatch(self, mini_batch, lr, lmbda, n):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [(1-lr*(lmbda/n))*w - (lr/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (lr/len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]
    
    def accuracy(self, data, convert = False):
        if convert:
            results = [(np.argmax(self.forward(x)), np.argmax(y)) for (x, y) in data]
        else:
            results = [(np.argmax(self.forward(x)), y) for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)
    
    def totalCost(self, data, lmbda, convert=False):
        cost = 0.0
        for x, y in data:
            a = self.forward(x)
            if convert:
                y = dst.vectorize(y)
            cost += self.cost.fn(a, y) / len(data)
        
        cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)
        return cost

    def save(self, filename):
        data = {
            'arch':self.arch,
            'weights':[w.tolist() for w in self.weights],
            'biases':[b.tolist() for b in self.biases],
            'cost':str(self.cost.__name__)
        }
        with open(filename, 'w') as f:
            json.dump(data, f)

def load(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    cost = getattr(sys.modules[__name__], data['cost'])
    net = Network(data['size'], cost_func = cost)
    net.weights = [np.array(w) for w in data['weights']]
    net.biases = [np.array(b) for b in data['biases']]
    return net