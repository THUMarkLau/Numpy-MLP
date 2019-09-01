import numpy as np 
import os 
import nn
import dataset as dts 
import matplotlib.pyplot as plt

def main():
    mnist = dts.MNIST('./')
    net = nn.Network([784, 300, 10], cost_func = nn.CrossEntropyCost)
    times = 80
    evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = \
    net.SGD(mnist.train_data, times, 10, 0.001, evaluation_data=mnist.test_data,
            monitor_evaluation_accuracy=True, monitor_training_cost=True, monitor_training_accuracy=True)
    temp = np.tile(100.0, times)
    evaluation_accuracy = evaluation_accuracy / temp
    x = np.arange(1, times+1, 1)
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax1 = plt.plot(x, evaluation_accuracy, 'g-', linewidth = 2)
    plt.xlabel('Epoch')
    plt.grid()

    ax2 = fig.add_subplot(2, 1, 2)
    ax2 = plt.plot(x, training_cost, 'r-', linewidth = 2)
    plt.ylabel('training_lost')
    plt.xlabel('Epoch')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()
