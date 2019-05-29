'''
cnn框架:
    tensorflow,为了加深对流程的理解，未使用高层的api keras
    手动实现了initial_parameters(),forward_propagation(),compute_cost,model()
结果：
 1. 数据量太小，始终会出现训练集准确率为1，而测试集合准确率只为0.5左右，不知道这种情况该如何平衡

注意：
1. tensorflow中cnn自动实现b的初始化，计算等工作，使用者只需要定义w
2. 困扰我很久的地方，循环读取训练图片放入numpy数组的时候，由于默认numpy数组初始化数据类型为float,这使得画图工具无法复现数组中某个图片
   而更奇葩的是，cv2画图要求图片数据为np.uint8类型，plt更宽松，只要是整形都能正常画出图片
3. pandas 读数据出现parse error, add error_bad_lines=False 发现是其中某一行多了个尾部空格导致的

'''
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from cnn_utils import load_dataset,random_mini_batches,convert_to_one_hot

np.random.seed(1)

def creat_placeholders(n_H,n_W,n_C,n_Y):
    X=tf.placeholder(dtype=tf.float32,shape=[None,n_H,n_W,n_C],name="X")
    Y=tf.placeholder(dtype=tf.float32,shape=[None,n_Y],name="Y")
    return X,Y

def initial_parameters():
    tf.set_random_seed(1)
    parameters={}
    parameters["W1"]=tf.get_variable(name="W1",shape=[4,4,3,1],initializer=tf.contrib.layers.xavier_initializer(seed = 0))
    parameters["W2"]=tf.get_variable(name="W2",shape=[2,2,1,1],initializer=tf.contrib.layers.xavier_initializer(seed = 0))
    return parameters

def forward_propagation(X,parameters,classes):
    W1=parameters["W1"]
    W2=parameters["W2"]

    Z1=tf.nn.conv2d(X,W1,strides=(1,1,1,1),padding="SAME")
    A1=tf.nn.relu(Z1)
    P1=tf.nn.max_pool(A1,ksize=(1,8,8,1),strides=(1,4,4,1),padding="VALID")

    #Z2=tf.nn.conv2d(P1,W2,strides=(1,1,1,1),padding="SAME")
    #A2=tf.nn.relu(Z2)
    #P2=tf.nn.max_pool(A2,ksize=(1,4,4,1),strides=(1,4,4,1),padding="SAME")
    P2=tf.contrib.layers.flatten(P1)
    #！！！！！here activation_fn must be None ,because loss and softmax are lumped together into a single function
    Z3=tf.contrib.layers.fully_connected(P2,classes,activation_fn=None)

    return Z3

def compute_cost(Z,Y):
    #tf.nn.sigmoid_cross_entropy_with_logits require (num,dim)
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z,labels=Y))
    return cost

def model(X_train,Y_train,X_test,Y_test,learning_rate=0.009,
            num_epochs=1000,minibatch_size=64,print_cost=True,classes=3):
    (m, n_H0, n_W0, n_C0)=X_train.shape
    d_y=Y_train.shape[1]
    #ops.reset_default_graph()
    tf.set_random_seed(1)
    seed=3
    costs=[]

    X,Y=creat_placeholders(n_H0, n_W0, n_C0,d_y)

    parameters=initial_parameters()

    Z=forward_propagation(X,parameters,classes)

    cost=compute_cost(Z,Y)

    optmizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init=tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_epochs):
            num_minibatches=int(m/minibatch_size)
            seed += 1
            minibatches=random_mini_batches(X_train, Y_train, minibatch_size, seed)
            epoch_cost=0
            for mbatch_x,mbatch_y in minibatches:
                _,minibatch_cost=sess.run([optmizer,cost],feed_dict={X:mbatch_x,Y:mbatch_y})
                epoch_cost += minibatch_cost/num_minibatches

            if print_cost == True and epoch % 10 == 0:
                costs.append(epoch_cost)
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))

        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        parameters=sess.run(parameters)

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z,1), tf.argmax(Y,1))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        max_index=tf.argmax(Z,1)
        print("argmax=",max_index.eval({X:X_test}))

        return parameters

#show data

X_train_orig,Y_train_orig,X_test_orig,Y_test_orig=load_dataset()
print("X_train_shape={},Y_train_shape={},X_test_shape={},Y_test_shape={}".format(X_train_orig.shape,Y_train_orig.shape,X_test_orig.shape,Y_test_orig.shape))
#fig,axes=plt.subplots()
#axes.imshow(X_train_orig[6])
#plt.show()

X_train=X_train_orig/255
X_test=X_test_orig/255
brand_type=3
Y_train = convert_to_one_hot(Y_train_orig,brand_type).T
Y_test=convert_to_one_hot(Y_test_orig,brand_type).T

parameters=model(X_train,Y_train,X_test,Y_test,num_epochs=60,minibatch_size=16,classes=brand_type)
