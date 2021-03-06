'''
author:huangchao
content:
1.linear regression realized by vector
2.code is organized by class
3.plot the origin data and fit curve

'''

import numpy as np
from matplotlib import pyplot as plt

class linear_regression_vectorized:
    def __init__(self):
        print(">>>>construct linear regression model")

    #config hyper parameters
    def fit(self,learning_rate=0.01,iter_num=100,batch_size=64):
        print(">>>>config hyper parameters")
        self.learning_rate=learning_rate
        self.iter_num=iter_num
        self.batch_size=batch_size

    def print_parameters(self):
        print("hyper parameters:learning_rate={},iter_num={},batch_size={}".format(self.learning_rate,self.iter_num,self.batch_size))
        print("construct weight:w={},b={}".format(self.construct_w,self.construct_b))

    #generate random linear data
    def gen_linear_dist_data(self):
        print(">>>>generate random linear data")
        np.random.seed(2)
        self.construct_w=np.random.randint(0,10)+np.random.random()
        self.construct_b=np.random.randint(0,5)+np.random.random()
        sample_num=100
        x_list=np.random.randint(0,100,[sample_num,1])*np.random.random()
        print(x_list)
        y_list=self.construct_w*x_list+self.construct_b+np.random.random([sample_num,1])*np.random.randint(-30,30,[sample_num,1])
        return x_list,y_list

    #train the model,update w,b and loss
    def train(self,x_list,y_list):
        print(">>>>train linear regression model")
        self.w=np.random.random()
        self.b=0
        sample_num=len(y_list)
        for i in range(self.iter_num):
            batch_index = np.random.choice(sample_num, self.batch_size)
            x_batch=x_list[batch_index]
            y_batch=y_list[batch_index]
            dw,db=self.calculate_gradient(x_batch,y_batch,self.w,self.b)
            self.update_parameter(dw,db)
            loss=self.calculate_loss(x_list,y_list,self.w,self.b)
            if(i%1000==0):
                print("After the {}th iterator:loss={},w={},b={},dw={},db={}".format(i,loss,self.w,self.b,dw,db))


    def calculate_gradient(self,x_batch,y_bacth,w,b):
        batch_size=len(y_bacth)
        pred_y=self.prediction(x_batch,w,b)
        dw = np.dot(x_batch.T,pred_y-y_bacth)
        db = np.sum(pred_y-y_bacth,keepdims=True)
        assert(dw.shape==(1,1))
        assert(db.shape==(1,1))
        return dw/batch_size,db/batch_size

    def prediction(self,x,w,b):
        y=x*w+b
        return y

    def update_parameter(self,dw,db):
        self.w -= self.learning_rate*dw
        self.b -= self.learning_rate*db

    def calculate_loss(self,x,y,w,b):
        num=len(x)
        average_loss = np.sum((w*x+b-y)**2)
        assert (average_loss.shape==())
        average_loss = average_loss/(2*num)
        return average_loss


#construct model
lr_model=linear_regression_vectorized()

#config model hyper parameters
lr_model.fit(learning_rate=0.001,iter_num=10000,batch_size=64)

#generate origin data
X,y=lr_model.gen_linear_dist_data()
lr_model.print_parameters()

#train the model ,update weight and loss
lr_model.train(X,y)

#plot origin data and fit curve
X_sort=X.copy()
X_sort.sort()
y_pred=X_sort*lr_model.w+lr_model.b
plt.figure(1)
plt.scatter(X,y,s=20,c='b')
plt.plot(np.squeeze(X_sort),y_pred,c='r')
plt.title("linear_regression_vectorized:origin data and fit data")
plt.show()


