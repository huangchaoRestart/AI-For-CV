import math
import numpy as np
import pandas as pd
import cv2

def load_dataset():
    # parse error, add error_bad_lines=False 发现是其中某一行多了个尾部空格导致的
    train_label = pd.read_csv("src_data/train/Label.TXT", header=None, sep=' ')
    train_brand = np.array(train_label.iloc[:, 1])
    train_num = len(train_brand)
    train_set_y_orig = train_brand.reshape((1, train_brand.shape[0]))

    test_label = pd.read_csv("src_data/test/Label.TXT", header=None, sep=' ')
    test_brand = np.array(test_label.iloc[:, 1])
    test_num = len(test_brand)
    test_set_y_orig = test_brand.reshape((1, test_brand.shape[0]))

    train_set_x_orig = np.zeros((train_num, 256, 256, 3), dtype=np.uint8)
    test_set_x_orig = np.zeros((test_num, 256, 256, 3), dtype=np.uint8)

    for i in range(train_num):
        path = "./src_data/Train/%d.jpg" % (i + 1)
        car_image = cv2.imread(path)
        #print("origin train pic size is {}".format(car_image.shape))
        car_image = cv2.resize(car_image, (256, 256))
        train_set_x_orig[i] = car_image

    for i in range(test_num):
        path = "./src_data/Test/%d.jpg" % (i + 1)
        car_image = cv2.imread(path)
        #print("origin test pic size is {}".format(car_image.shape))
        car_image = cv2.resize(car_image, (256, 256))
        test_set_x_orig[i] = car_image
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y
