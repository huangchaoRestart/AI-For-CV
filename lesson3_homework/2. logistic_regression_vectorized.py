'''
author:huangchao
content:
1.logistic regression realized by vector
2.code is organized by class
3.calculate accuracy of train set
'''

import numpy as np

class logistic_regression_vectorized:
    def __init__(self):
        print(">>>>construct logistic regression model")

    #config hyper parameters
    def fit(self,learning_rate=0.01,iter_num=100,batch_size=64):
        print(">>>>config hyper parameters")
        self.learning_rate=learning_rate
        self.iter_num=iter_num
        self.batch_size=batch_size

    def print_parameters(self):
        print("hyper parameters:learning_rate={},iter_num={},batch_size={}".format(self.learning_rate,self.iter_num,self.batch_size))

    #generate random linear data
    def gen_linear_dist_data(self):
        print(">>>>generate random linear data")
        #np.random.seed(2)
        sample_num=100
        x_list_0=np.random.randint(0,55,[sample_num,1])
        x_list_1 = np.random.randint(45, 100, [sample_num, 1])
        y_list_0=np.zeros([sample_num,1])
        y_list_1=np.ones([sample_num,1])
        return np.row_stack((x_list_0,x_list_1)),np.row_stack((y_list_0,y_list_1))

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
        pred_y=self.sigmoid(x_batch,w,b)
        dw = np.dot(x_batch.T,pred_y-y_bacth)
        db = np.sum(pred_y-y_bacth,keepdims=True)
        assert(dw.shape==(1,1))
        assert(db.shape==(1,1))
        return dw/batch_size,db/batch_size

    def sigmoid(self,x,w,b):
        y=1/(1+np.exp(-x*w-b))
        return y

    def update_parameter(self,dw,db):
        self.w -= self.learning_rate*dw
        self.b -= self.learning_rate*db

    def calculate_loss(self,x,y,w,b):
        num=len(x)
        average_loss = np.dot(y.T,np.log10(self.sigmoid(x,w,b)))+np.dot((1-y).T,np.log10(1-self.sigmoid(x,w,b)))
        average_loss = -average_loss/num
        assert (average_loss.shape==(1,1))
        return average_loss


#construct model
lr_model=logistic_regression_vectorized()

#config model hyper parameters
lr_model.fit(learning_rate=0.01,iter_num=50000,batch_size=64)

#generate origin data
X,y=lr_model.gen_linear_dist_data()
lr_model.print_parameters()

#train the model ,update weight and loss
lr_model.train(X,y)

#calculate accurate of training set
train_num=len(y)
y_pred=lr_model.sigmoid(X,lr_model.w,lr_model.b)
y_pred[y_pred>0.5]=1
y_pred[y_pred<=0.5]=0
accurate=train_num-np.sum(np.abs(y-y_pred),keepdims=False)
print("The accuracy on training set is {}%".format(accurate/train_num*100))


