'''author:huangchao  created date=20190405  class:image transform'''

import cv2
import numpy as np
from matplotlib import pyplot as plt


class image_transform:
    def __init__(self, path):
        self.image = cv2.imread(path)

    def plot(self, name, image):
        cv2.imshow(name, image)
        key = cv2.waitKey()
        if key == 27:
            cv2.destroyAllWindows()

    def crop(self, image, row_ratio=1 / 2, col_ratio=1 / 2):
        self.image_cropped = image[0:int(image.shape[0] * row_ratio), 0:int(image.shape[1] * col_ratio)]
        print("crop shape:{}".format(self.image_cropped.shape))
        return self.image_cropped

    def split(self, image):
        self.B, self.G, self.R = cv2.split(image)
        print("B shape:{}, G shape:{}".format(self.B.shape, self.G.shape))
        return self.B, self.G, self.R

    def change_color(self, image_level, upper=50):
        self.image_level = image_level
        filter_th = np.random.randint(-upper, upper)
        if 0 == filter_th:
            pass
        elif filter_th > 0:
            right_th = 255 - filter_th
            self.image_level[self.image_level > right_th] = 255
            self.image_level[self.image_level <= filter_th] = (
                    filter_th + self.image_level[self.image_level <= filter_th]).astype(
                self.image_level.dtype)
        elif filter_th < 0:
            left_th = 0 - filter_th
            self.image_level[self.image_level < left_th] = 0
            self.image_level[self.image_level >= left_th] = (
                    filter_th + self.image_level[self.image_level >= left_th]).astype(
                self.image_level.dtype)
        return self.image_level

    def adjust_gamma(self, image, gamma=1):
        gamma_inv = 1 / gamma
        table = []
        for i in range(256):
            table.append(((i / 255.0) ** gamma_inv) * 255)
        table = np.array(table).astype("uint8")
        self.image_adjust = cv2.LUT(image, table)
        return self.image_adjust

    def histogram(self, image, channel=0):
        # img_small_brighter = cv2.resize(image_brighter, (int(image_brighter.shape[0]*0.5), int(image_brighter.shape[1]*0.5)))
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        # equalize the histogram of the Y channel
        img_yuv[:, :, channel] = cv2.equalizeHist(img_yuv[:, :, channel])  # only for 1 channel
        # convert the YUV image back to RGB format
        self.img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return self.img_output

    def rotation(self, image, center=(0, 0), angle=30, scale=0.5):
        M = cv2.getRotationMatrix2D(center, angle, scale)  # center, angle, scale
        self.img_rotate = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        return self.img_rotate

    def affine(self, image, origin, trans):
        # Affine Transform
        M = cv2.getAffineTransform(origin, trans)
        self.image_affine = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        return self.image_affine

    def random_warp(self, image, upper=120):
        height, width, channels = image.shape
        # warp:
        random_margin = upper
        x1 = np.random.randint(-random_margin, random_margin)
        y1 = np.random.randint(-random_margin, random_margin)
        x2 = np.random.randint(width - random_margin - 1, width - 1)
        y2 = np.random.randint(-random_margin, random_margin)
        x3 = np.random.randint(width - random_margin - 1, width - 1)
        y3 = np.random.randint(height - random_margin - 1, height - 1)
        x4 = np.random.randint(-random_margin, random_margin)
        y4 = np.random.randint(height - random_margin - 1, height - 1)

        dx1 = np.random.randint(-random_margin, random_margin)
        dy1 = np.random.randint(-random_margin, random_margin)
        dx2 = np.random.randint(width - random_margin - 1, width - 1)
        dy2 = np.random.randint(-random_margin, random_margin)
        dx3 = np.random.randint(width - random_margin - 1, width - 1)
        dy3 = np.random.randint(height - random_margin - 1, height - 1)
        dx4 = np.random.randint(-random_margin, random_margin)
        dy4 = np.random.randint(height - random_margin - 1, height - 1)

        pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        print("perspective M={}".format(M))
        self.img_warp = cv2.warpPerspective(image, M, (width, height))
        return self.img_warp


path = "pic/lesson1/doggy.jpg"
transform = image_transform(path)
# crop
image_cropped = transform.crop(transform.image, 2 / 3, 2 / 3)
transform.plot("croped pic", image_cropped)
# split
B, G, R = transform.split(transform.image)
transform.plot("B pic", B)
# change color by operate pixel
B_change_color = transform.change_color(B, upper=50)
G_change_color = transform.change_color(G, upper=50)
R_change_color = transform.change_color(R, upper=50)
image_merge = cv2.merge((B_change_color, G_change_color, R_change_color))
transform.plot("color changed pic", image_merge)
# gamma adjust
image_brigter = transform.adjust_gamma(transform.image, gamma=2)
transform.plot("image_brigter pic", image_brigter)
# histogram equalization
image_hist = transform.histogram(image_brigter, channel=1)
transform.plot("image_hist pic", image_hist)
# rotate
image_rotated = transform.rotation(transform.image, center=(0, 0), angle=30, scale=0.5)
transform.plot("image_rotated pic", image_rotated)
# affine
pts1 = np.float32([[0, 0], [transform.image.shape[1] - 1, 0], [0, transform.image.shape[0] - 1]])
pts2 = np.float32([[0, 0], [transform.image.shape[1] * 0.5, 0], [0, transform.image.shape[0] * 0.5]])
image_affine = transform.affine(transform.image, origin=pts1, trans=pts2)
transform.plot("image_affine pic", image_affine)
# perspective
image_warp = transform.random_warp(transform.image, upper=120)
transform.plot("image_warp pic", image_warp)
