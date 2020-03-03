# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import cv2
import numpy as np
from pse import pse
from model_net import psenet_model as PM
from model_net import model_v1
from data_pre import data_pre,pre_tools
os.environ['CUDA_VISIBLE_DEVICES']='0'

flages=tf.flags
FLAGS = flages.FLAGS
flages.DEFINE_integer("kernal_num", 3, "kernal number")
flages.DEFINE_float("min_area", 800.0, "min_area")
flages.DEFINE_integer("scale", 1, "scale ")
flages.DEFINE_float("min_kernel_area", 5.0, "min kernel area ")
flages.DEFINE_float("min_score", 0.93, "min scale ")


def Test():
	test_data_load = data_pre.DataTest_load_pre(long_size = 320)
	print("Data num: ",len(test_data_load))

	tf_image = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 3], name="image")

	#############################################################################################
	###  Model logites And Model Path
	###  Self Model
	#resnet = PM.ResNet(PM.BottleBlock(), FLAGS.kernal_num, True, 1.0)
	#logites = resnet(tf_image)  ## (batch, 7, size, size)
	#model_path = "./checkpoints/old/PSENet_BC-32_k3_2020-03-02-19-31-31.ckpt-192500"
	### Model two
	logites, _ = model_v1.model(tf_image, FLAGS.kernal_num) ## [1,3,?,?]
	model_path = "./checkpoints/0302/PSENet_BC-32_k3_2020-02-26-17-06-44.ckpt-192500"
	#############################################################################################

	saver = tf.train.Saver()
	sess_config = tf.ConfigProto(allow_soft_placement=True)
	sess_config.gpu_options.allow_growth = True
	sess = tf.Session(config=sess_config)

	saver.restore(sess=sess, save_path=model_path)
	print("################ load model down! ##########################")

	for i in range(len(test_data_load)):
		ori_img, scaled_img = test_data_load[i] ### (h,w,3)

		text_box = ori_img.copy()
		scaled_img = np.expand_dims(scaled_img,axis=0) ### (1, h, w, 3)

		train_pred = sess.run([logites],feed_dict={tf_image: scaled_img})  ## [(1, 7, size, size)]
		train_pred = train_pred[0]  ### (1, k, h, w) 0~1之间
		#train_score = (pre_tools.sigmod(train_pred[0,0,:,:])).astype(np.float32) ## [512,512]

		mask = train_pred[:,0,:,:] ## [1,512,512] ## 取第一个 kernal 作为 mask
		kernels = train_pred[:,0:,:,:] * mask  ## [1,3,512,512] 对后kernal进行mask处理
		kernels = np.squeeze(kernels,0).astype(np.uint8)  ##  [3,512,512]

		### pse 渐进扩展输出
		pred = pse(kernels,FLAGS.min_kernel_area/(FLAGS.scale * FLAGS.scale))
		#cv2.imwrite("./Images/Image_OUT/image_Pred_2{}.jpg".format(i), pred * 255) ## 输出最终结果

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
			rect = cv2.minAreaRect(points)
			bbox = cv2.boxPoints(rect) * scale
			bbox = bbox.astype('int32')
			bboxes.append(bbox.reshape(-1))

		for bbox in bboxes:
			cv2.drawContours(text_box, [bbox.reshape(4, 2)], -1, (0, 255, 0), 1)

		text_box = cv2.resize(text_box, (ori_img.shape[1], ori_img.shape[0]))

		cv2.imwrite("./Images/Image_OUT/img_{}.jpg".format(i), text_box)
		pre_tools.write_result_as_txt(str(i),bboxes,'./Images/Text_OUT/')
		print("Finish {} image!".format(i+1))


if __name__=="__main__":
	Test()
