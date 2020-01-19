# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import cv2
import numpy as np
import time
import logging
from pse import pse
from model_net import psenet_model as PM
from data_pre import data_pre,pre_tools
os.environ['CUDA_VISIBLE_DEVICES']='0'

flages=tf.flags
FLAGS = flages.FLAGS
flages.DEFINE_integer("kernal_num", 3, "kernal number")
flages.DEFINE_float("min_area", 800.0, "min_area")
flages.DEFINE_integer("scale", 1, "scale ")
flages.DEFINE_float("min_score", 0.93, "scale ")
flages.DEFINE_boolean("model_restore",False,"wheater to restore model")


def Test():
	test_data_load = data_pre.DataTest_load_pre()
	print("Data num: ",len(test_data_load))

	tf_image = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 3], name="image")

	resnet = PM.ResNet(PM.BottleBlock(), FLAGS.kernal_num, True, 1.0)
	logites = resnet(tf_image)  ## (batch, k, size, size)

	saver = tf.train.Saver()
	sess_config = tf.ConfigProto(allow_soft_placement=True)
	sess_config.gpu_options.allow_growth = True
	sess = tf.Session(config=sess_config)

	model_path = "./checkpoints/PSENet_BC-32_2020-01-15-10-18-40.ckpt-192500"
	saver.restore(sess=sess, save_path=model_path)
	print("################ load model down! ##########################")

	bias = 1.0
	for i in range(len(test_data_load)):
		ori_img, scaled_img = test_data_load[i] ### (h,w,3)
		scaled_img = np.expand_dims(scaled_img,axis=0) ### (1, h, w, 3)

		train_pred = sess.run([logites],feed_dict={tf_image: scaled_img})  ## [(1, 7, size, size)]
		#train_score = train_score[0]
		train_pred = train_pred[0]  ### (1, k, h, w)
		train_score = (pre_tools.sigmod(train_pred[0,0,:,:])).astype(np.float32)

		text_box = ori_img.copy()

		outputs = (np.sign(train_pred - bias) + 1) / 2  ### 确定在 (0, 1) 之间

		###  因为这里 batch 为 1 ， 故 outputs 第一维度就为 0
		train_text_img  = outputs[:, 0,:,:] ## (1, h, w)
		train_kernals = outputs[:, 0:, :, :] * train_text_img  ## (1, 7, h, w)

		train_kernals =train_kernals[0].astype(np.uint8) ## (7, h, w)

		pred = pse(kernals=train_kernals, min_area = 5.0)
		#cv2.imwrite("./Images/Image_OUT/image_Pred{}.jpg".format(i), pred*255)

		scale = (ori_img.shape[1] * 1.0 / pred.shape[1], ori_img.shape[0] * 1.0 / pred.shape[0])  ## 变换尺寸
		label = pred
		label_num = np.max(label) + 1
		bboxes = []
		for j in range(1, label_num):
			#point_where = np.where(label == 1)
			try:
				points = np.array(np.where(label == j)).transpose((1, 0))[:, ::-1]
			except:
				continue

			if points.shape[0] < FLAGS.min_area / (FLAGS.scale * FLAGS.scale):
				continue

			score_j = np.mean(train_score[label == j])
			if score_j < FLAGS.min_score:
				continue

			rect = cv2.minAreaRect(points)
			bbox = cv2.boxPoints(rect) * scale
			bbox = bbox.astype('int32')
			bboxes.append(bbox.reshape(-1))

		for bbox in bboxes:
			cv2.drawContours(text_box, [bbox.reshape(4, 2)], -1, (0, 255, 0), 1)

		text_box = cv2.resize(text_box, (ori_img.shape[1], ori_img.shape[0]))

		cv2.imwrite("./Images/Image_OUT/img_{}.jpg".format(i), text_box)
		#print("Stop!")


if __name__=="__main__":
	Test()
