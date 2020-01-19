# -*- coding:utf-8 -*-
import os
import tensorflow as tf
import numpy as np
import random
from data_pre import pre_tools
import cv2

### training data
image_path = "X:/fffan/Data/OCR_Detection/ICPR_text_train_20180313/try_img/"
gt_path = "X:/fffan/Data/OCR_Detection/ICPR_text_train_20180313/try_txt/"

### test data
test_image_path = "E:/fffan/Detec/PSEnet/PSEnet_tensorflow/Images/Image_IN/"

def get_data_path(image_path=image_path,gt_path=gt_path):
	img_names = os.listdir(image_path)
	data_path = []
	for idx, img_name in enumerate(img_names):
		image = os.path.join(image_path,img_name)
		#print(img_name.split(".jpg"))
		gt_name = img_name.split(".jpg")[0] + ".txt"
		gt = os.path.join(gt_path,gt_name)
		if os.path.exists(gt):
			data_path.append([image,gt])
	return data_path

class Data_load_pre(object):
	def __init__(self,data_path,kernal_num,min_scale=0.4):
		self.kernal_num = kernal_num
		self.min_scale = min_scale
		self.size = 320  ####resize
		random.shuffle(data_path)
		self.data_path = data_path

	def __len__(self):
		return len(self.data_path)

	def __getitem__(self, item):
		self.image = self.data_path[item][0]
		self.gt = self.data_path[item][1]

		img = pre_tools.read_image(self.image)
		gt_box,gt_tags = pre_tools.get_box(self.gt,img)

		########################################################
		###  该处进行 img 尺寸调整。
		img = cv2.resize(img, (self.size, self.size))
		########################################################
		new_gt_box = gt_box * ([img.shape[1],img.shape[0]]*4)

		gt_text = np.zeros(img.shape[0:2], dtype="uint8")
		train_mask = np.ones(img.shape[0:2], dtype="uint8")

		if new_gt_box.shape[0] > 0: ### (n,8)
			new_gt_box = new_gt_box.reshape((new_gt_box.shape[0],gt_box.shape[1]//2,2)).astype('int32') ## (n,8) ==> (n,4,2)
			for i in range(new_gt_box.shape[0]):
				cv2.drawContours(gt_text,[new_gt_box[i]],-1,i+1,-1)## 填充轮廓颜色
				### 对反例进行标注：mask为False
				###  反例：即是文本，但是模糊不清，无法看出具体内容
				if not gt_tags[i]:
					cv2.drawContours(train_mask,[new_gt_box[i]],-1, 0,-1) ## 对反例进行标0

		gt_text[gt_text > 0] = 1


		kernals_image = []
		for i in range(1,self.kernal_num): ## k-1 个
			kernal_img = np.zeros(img.shape[0:2], dtype="uint8")
			rate = 1.0 - (1.0 - self.min_scale)/(self.kernal_num - 1) * i
			new_boxes = pre_tools.box_shrink(new_gt_box,rate)
			#kernals_boxes.append(new_boxes)
			for i in range(new_boxes.shape[0]):
				cv2.drawContours(kernal_img, [new_boxes[i]], -1, i+1, -1)
			kernal_img = kernal_img.reshape((1, kernal_img.shape[0], kernal_img.shape[1])).astype('uint8')
			kernals_image.append(kernal_img)

		kernals_image = np.concatenate(kernals_image, 0) ### concat

		##############################################################
		###  对图像进行 旋转，翻转，亮度、对比度等调节
		###  略
		##############################################################
		img_normalize = np.zeros(img.shape[0:2],dtype=np.float32)
		img_normalize = cv2.normalize(img,img_normalize,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

		return img_normalize,gt_text,train_mask,kernals_image

class DataTest_load_pre(object):
	def __init__(self, long_size=1024):
		img_names = os.listdir(test_image_path)
		self.img_paths = []
		for idx, img_name in enumerate(img_names):
			image_path = os.path.join(test_image_path, img_name)
			self.img_paths.append(image_path)
		self.long_size = long_size

	def __len__(self):
		return len(self.img_paths)

	def __getitem__(self, index):
		img_path = self.img_paths[index]

		ori_img = pre_tools.read_image(img_path)
		scaled_img = pre_tools.scale(ori_img,long_size=512)
		#scaled_img = pre_tools.scale(ori_img, self.long_size)  # 尺寸放大
		#scaled_img = cv2.resize(ori_img, (self.long_size, self.long_size))

		img_normalize = np.zeros(scaled_img.shape[0:2], dtype=np.float32)
		img_normalize = cv2.normalize(scaled_img, img_normalize, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

		return ori_img, img_normalize
