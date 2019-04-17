'''
作者：黄超
时间:20190415
说明：该代码并未经debug测试，只是实现ransac思路
实现步骤：
1.  随机选择4组匹配点，并且确保任意三点不在一条直线。
    判断共线的方法采用三点组成的三角形面积>0,而面积的计算可以直接通过计算三点组成的向量的行列式实现
    函数定义：def check4points()
2.  根据4组匹配点确定投影矩阵
3.  根据计算好的投影矩阵计算所有局外点的方差，选择方差满足门限threshold的点作为新加入的局内点，统计局内点个数
    函数定义：def findInliers()
4.  基于选择的所有局内点，根据最小二乘法重新计算投影矩阵
    函数定义：def computePerspectiveM()
5.  重复步骤1-4，直到迭代次数达到k，则停止，并从历史结果中选择最优的投影矩阵(局内点最多)
    其中随机选点的迭代次数K 可根据期望正确概率以及内点比例计算得出
'''

import numpy as np
import cv2

#subset:list of list 3 points in image
def check3points(subset):
    #计算3点组成的两个向量的行列式判断是否共线
    (a1,a2)=subset[0][0],subset[0][1]
    (b1,b2)=subset[1][0],subset[1][1]
    (c1,c2)=subset[2][0],subset[2][1]
    vec1=(b1-a1,b2-a2)
    vec2=(c1-a1,c2-a2)
    m=np.vstack((vec1,vec2))
    m_det=np.linalg.det(m)
    return m_det

#src:list of list 4 points in image
def check4points(src):
    #依次从4点中选出3点判断是否共线
    for i in range(len(src)):
        temp=src.copy()
        det=check3points(np.delete(temp,i,0))
        if det<1e-8:
            return False
    return True

#计算每对匹配点的投影误差，并根据门限判断其是否为内点
def findInliers(A,B,perM,threshold):
    pnum=len(A)
    inlinerIndex=[]
    count=0
    for i in range(pnum):
        denominator=perM[2][0]*A[i][0]+perM[2][1]*A[i][1]+perM[2][2]
        dx=(perM[0][0]*A[i][0]+perM[0][1]*A[i][1]+perM[0][2])/denominator-B[i][0]
        dy=(perM[1][0]*A[i][0]+perM[1][1]*A[i][1]+perM[1][2])/denominator-B[i][1]
        error=dx*dx+dy*dy
        if(error<threshold):
            count += 1
            inlinerIndex.append(i)
    return inlinerIndex,count

#根据最小二乘法寻找最佳的投影矩阵，解最小二乘法的方法选择梯度下降法
def computePerspectiveM(A,B, iterNum=200, learning_rate=0.01):
    pnum=A.shape[0]
    #随机生成初始投影矩阵
    h0=np.random.rand(1,3)
    h1 = np.random.rand(1, 3)
    h2 = [np.random.rand(1, 3)]
    h2[0,-1]=1
    A=np.column_stack(A,np.ones(pnum))
    for i in range(iterNum):
        xhypoth=np.sum(np.multiply(h0,A),axis=1)/np.sum(np.multiply(h2,A),axis=1)
        yhypoth=np.sum(np.multiply(h1,A),axis=1)/np.sum(np.multiply(h2,A),axis=1)
        dh00=np.sum(np.multiply(xhypoth-A[:,0],A[:,0]))/pnum
        dh01= np.sum(np.multiply(xhypoth-A[:,0], A[:, 1])) / pnum
        dh02= np.sum(np.multiply(xhypoth-A[:,0], 1)) / pnum
        dh10=np.sum(np.multiply(yhypoth-B[:,1], A[:, 0])) / pnum
        dh11=np.sum(np.multiply(yhypoth-B[:,1], A[:, 0])) / pnum
        dh12=np.sum(np.multiply(yhypoth-B[:,1], 1)) / pnum
        dh20=np.sum(np.multiply(xhypoth-A[:,0], np.multiply(A[:, 0],-xhypoth/np.sum(np.multiply(h2,A),axis=1)))) / pnum
        dh21=np.sum(np.multiply(xhypoth-A[:,0], np.multiply(A[:, 1],-xhypoth/np.sum(np.multiply(h2,A),axis=1)))) / pnum
        h0[0] -= learning_rate*dh00
        h0[1] -= learning_rate * dh01
        h0[2] -= learning_rate * dh02
        h1[0] -= learning_rate * dh10
        h1[1] -= learning_rate * dh11
        h1[2] -= learning_rate * dh12
        h2[0] -= learning_rate * dh20
        h2[1] -= learning_rate * dh21
    return np.array((h0,h1,h2))

def ransacMatching(A, B, K=30, thres=3,learning_rate=0.01,iterNum=200):
    A=np.array(A)
    B=np.array(B)
    maxInlinerNum=0
    for k in range(K):
        #随机选择4对点
        randomPointIndex=np.random.choice(len(A),4)
        imageSource4=A[randomPointIndex]
        imageDest4=B[randomPointIndex]

        #判断4点是否共线
        if check4points(imageSource4)==False or check4points(imageDest4)==False:
            continue

        #根据4对匹配点计算投影矩阵
        M = cv2.getPerspectiveTransform(np.float32(imageSource4), np.float32(imageDest4))
        print("perspective M={}".format(M))

        #根据投影矩阵计算每对匹配点的投影误差，并判断是否为内点
        inlinerIndex,inlinerNum=findInliers(A,B,M,thres)

        #如果内点大于历史最佳内点数，重新计算所有内点的投影矩阵
        if inlinerNum>maxInlinerNum:
            maxInlinerNum=inlinerNum
            bestM=computePerspectiveM(A[inlinerIndex],B[inlinerIndex],iterNum,learning_rate)

    #返回最佳投影矩阵
    return bestM
