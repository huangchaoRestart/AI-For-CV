import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.framework import ops
from tf_utils import load_dataset,random_mini_batches,convert_to_one_hot,predict

def creat_placeholders(d_x,d_y):
    X=tf.placeholder(dtype=tf.float32,shape=[d_x,None],name="X")
    Y=tf.placeholder(dtype=tf.float32,shape=[d_y,None],name="Y")
    return X,Y

def initial_parameters(layers):
    tf.set_random_seed(1)
    num=len(layers)
    parameters={}
    for i in range(1,num):
        parameters["W"+str(i)]=tf.get_variable(name="W"+str(i),shape=[layers[i],layers[i-1]],initializer=tf.contrib.layers.xavier_initializer(seed = 1))
        parameters["b"+str(i)]=tf.get_variable(name="b"+str(i),shape=[layers[i],1],initializer=tf.zeros_initializer())
    return parameters

def forward_propagation(X,parameters):
    num=len(parameters)//2
    A_prev=X
    for i in range(1,num+1):
        Z=tf.add(tf.matmul(parameters["W"+str(i)],A_prev),parameters["b"+str(i)])
        A=tf.nn.relu(Z)
        A_prev=A

    return Z

def compute_cost(Z,Y):
    #tf.nn.sigmoid_cross_entropy_with_logits require (num,dim)
    log=tf.transpose(Z)
    lab=tf.transpose(Y)
    cost=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=log,labels=lab))
    return cost

def model(X_train,Y_train,X_test,Y_test,layers,learning_rate=0.0001,
            num_epochs=200,minibatch_size=32,print_cost=True):
    (d_x,num_pics)=X_train.shape
    d_y=Y_train.shape[0]
    #ops.reset_default_graph()
    tf.set_random_seed(1)
    seed=3
    costs=[]

    X,Y=creat_placeholders(d_x,d_y)

    parameters=initial_parameters(layers)

    Z=forward_propagation(X,parameters)

    cost=compute_cost(Z,Y)

    optmizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init=tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_epochs):
            num_minibatches=int(num_pics/minibatch_size)
            seed += 1
            minibatches=random_mini_batches(X_train, Y_train, minibatch_size, seed)
            epoch_cost=0
            for mbatch_x,mbatch_y in minibatches:
                _,minibatch_cost=sess.run([optmizer,cost],feed_dict={X:mbatch_x,Y:mbatch_y})
                epoch_cost += minibatch_cost/num_minibatches

            if print_cost == True and epoch % 10 == 0:
                costs.append(epoch_cost)
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
                #print(mbatch_x[0,:])

        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        parameters=sess.run(parameters)

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters


#show data
X_train_orig,Y_train_orig,X_test_orig,Y_test_orig,classes=load_dataset()
print("X_train_shape={},Y_train_shape={},X_test_shape={},Y_test_shape={}".format(X_train_orig.shape,Y_train_orig.shape,X_test_orig.shape,Y_test_orig.shape))
#fig,axes=plt.subplots()
#axes.imshow(X_train_orig[0])
#plt.show()

X_train_flatten=X_train_orig.reshape(X_train_orig.shape[0],-1).T
X_test_flatten=X_test_orig.reshape(X_test_orig.shape[0],-1).T
X_train=X_train_flatten/255
X_test=X_test_flatten/255
Y_train = convert_to_one_hot(Y_train_orig,6)
Y_test=convert_to_one_hot(Y_test_orig,6)
print(X_train.shape+Y_train.shape)

layers=[12288,25,12,6]
parameters=model(X_train,Y_train,X_test,Y_test,layers)
