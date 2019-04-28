'''
author：huangchao
methods:
1. read two pics and extract sift keypoints and descriptors
2. find the matched keypoints by knn and descriptors
3. find the homography matrix by ransac
4. stitch the two pics directly(however it would be better to stitch two pics according to the weights of distance)
'''

import cv2
import numpy as np

class image_stitching:
    #读取待拼接的两张图片
    def __init__(self,path_left="pic/left.jpeg",path_right="pic/right.jpeg",show_pic=True):
        self.image_left=cv2.imread(path_left)
        self.image_right=cv2.imread(path_right)
        if show_pic==True:
            print("shape of image_left and image_right is {},{}".format(self.image_left.shape,self.image_right.shape))
            cv2.imshow("left",self.image_left)
            cv2.imshow("right",self.image_right)
            if cv2.waitKey()==27:
                cv2.destroyAllWindows()

    #分别提取两张图片的sift特征点和描述子
    def sift_detect(self,show_pic=True):
        sift=cv2.xfeatures2d.SIFT_create()
        (self.kp_left,self.des_left)=sift.detectAndCompute(self.image_left,None)
        (self.kp_right,self.des_right)=sift.detectAndCompute(self.image_right,None)
        if show_pic==True:
            image_left_sift = cv2.drawKeypoints(self.image_left, self.kp_left, outImage=np.array([]),
                                                flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
            image_right_sift = cv2.drawKeypoints(self.image_right, self.kp_right, outImage=np.array([]),
                                                 flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
            print("the descriptor shape of left and right is {},{}".format(self.des_left.shape,self.des_right.shape))
            cv2.imshow("left_sift",image_left_sift)
            cv2.imshow("right_sift",image_right_sift)
            if cv2.waitKey() == 27:
                cv2.destroyAllWindows()

    #用KD tree实现特征点匹配
    def sift_match(self,ratio=0.5,show_pic=True):
        FLANN_INDEX_KDTREE = 0  # kd树
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(self.des_left, self.des_right, k=2)
        # store all the good matches as per Lowe's ratio test
        self.match_filtered = []
        self.kp_left_filtered_idx=[]
        self.kp_right_filtered_idx=[]
        for m, n in matches:
            if m.distance < ratio * n.distance:
                self.match_filtered.append(m)
                self.kp_left_filtered_idx.append(m.queryIdx)
                self.kp_right_filtered_idx.append(m.trainIdx)
        if show_pic==True:
            kp_left_filtered = [self.kp_left[i] for i in self.kp_left_filtered_idx]
            kp_right_filtered = [self.kp_right[i] for i in self.kp_right_filtered_idx]
            image_left_sift_filtered = cv2.drawKeypoints(self.image_left, kp_left_filtered, outImage=np.array([]),color=(0,0,255),
                                                flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
            image_right_sift_filtered = cv2.drawKeypoints(self.image_right, kp_right_filtered, outImage=np.array([]),color=(0,0,255),
                                                 flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
            print("num of filtered matched sift is {}".format(len(self.match_filtered)))
            cv2.imshow("left_sift_filtered",image_left_sift_filtered)
            cv2.imshow("right_sift_filtered",image_right_sift_filtered)
            if cv2.waitKey() == 27:
                cv2.destroyAllWindows()

        return len(self.match_filtered)

    #根据匹配后的SIFT特征点计算单应性矩阵
    def find_homography(self,match_num,show_pic=True):
        if(match_num<10):
            print("no enough matched sift found")
        src_pts = np.array([self.kp_left[m.queryIdx].pt for m in self.match_filtered]).reshape(-1, 1, 2)
        dst_pts = np.array([self.kp_right[m.trainIdx].pt for m in self.match_filtered]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h,w = self.image_left.shape[:2]
        h1,w1=self.image_right.shape[:2]
        shft = np.array([[1, 0, w], [0, 1, 0], [0, 0, 1]])
        M = np.dot(shft, H)  # 获取左边图像到右
        self.image_left_pers = cv2.warpPerspective(self.image_left, M, (w*2, h))
        self.image_left_pers[0:h,w:2*w]=self.image_right
        self.image_left_pers=cv2.resize(self.image_left_pers,(w+500,h))

        if show_pic==True:
            #cv2.imshow("image_left", self.image_left)
            #cv2.imshow("image_right", self.image_right)
            cv2.imshow("image_stitching", self.image_left_pers)
            if cv2.waitKey() == 27:
                cv2.destroyAllWindows()



if __name__=="__main__":
    image_st=image_stitching(show_pic=False)
    image_st.sift_detect(show_pic=True)
    match_num=image_st.sift_match(ratio=0.5,show_pic=True)
    image_st.find_homography(match_num,show_pic=True)
