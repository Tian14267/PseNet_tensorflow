# -*- coding: utf-8 -*-
import tensorflow as tf
#####################################
###  tensorflow 2.0以上最新写法
import os
import time
import logging
import numpy as np
from metrics import runningScore
from model_net import psenet_model_3 as PM
from data_pre import data_pre,pre_tools
from tensorflow.python.platform import flags
os.environ['CUDA_VISIBLE_DEVICES']='0'
#tf.enable_eager_execution()  ### tensorflow 2.0 以上就不需要了

FLAGS = flags.FLAGS
flags.DEFINE_float("learning_rate",0.001,"learning rate")
flags.DEFINE_integer("kernal_num", 3, "kernal number")
flags.DEFINE_boolean("model_restore",False,"wheater to restore model")
flags.DEFINE_integer("batch_size", 32, "batch size")
flags.DEFINE_integer("epochs", 800, "epochs of net")

def cal_text_score(texts, gt_texts, training_masks, running_metric_text):
	training_masks = training_masks.numpy()
	pred_text = tf.sigmoid(texts).numpy() * training_masks
	pred_text[pred_text <= 0.5] = 0
	pred_text[pred_text > 0.5] = 1
	pred_text = pred_text.astype(np.int32)
	gt_text = gt_texts.numpy() * training_masks
	gt_text = gt_text.astype(np.int32)
	running_metric_text.update(gt_text, pred_text)
	score_text, _ = running_metric_text.get_scores()
	return score_text

def cal_kernel_score(kernels, gt_kernels, gt_texts, training_masks, running_metric_kernel):
	mask = (gt_texts * training_masks).numpy()
	kernel = kernels[:, -1, :, :]
	gt_kernel = gt_kernels[:, -1, :, :]
	# pred_kernel = torch.sigmoid(kernel).data.cpu().numpy()
	pred_kernel = tf.sigmoid(kernel).numpy()
	pred_kernel[pred_kernel <= 0.5] = 0
	pred_kernel[pred_kernel > 0.5] = 1
	pred_kernel = (pred_kernel * mask).astype(np.int32)
	gt_kernel = gt_kernel.numpy()
	gt_kernel = (gt_kernel * mask).astype(np.int32)
	running_metric_kernel.update(gt_kernel, pred_kernel)
	score_kernel, _ = running_metric_kernel.get_scores()
	return score_kernel

def train(data_loader,model,epoch,num_batch,learning_rate):
	####  Model One
	#resnet = PM.ResNet(PM.BottleBlock(), FLAGS.kernal_num, True, 0.5)
	#logites = resnet(tf_image)  ## (batch, 7, size, size)

	#Loss = PM.Dec_Loss_2(logites=logites, gt_texts=tf_gt, gt_kernels=tf_kernal, training_masks=tf_mask)

	running_metric_text = runningScore(2)
	running_metric_kernel = runningScore(2)

	'''
	learning_rate = tf.train.exponential_decay(
		learning_rate=FLAGS.learning_rate,
		global_step= step,
		decay_steps=num_batch * 200,
		decay_rate=0.5,
		staircase=True)
	'''


	#optim = tf.train.AdamOptimizer(learning_rate=learning_rate)
	optim = tf.optimizers.Adam(learning_rate=learning_rate)

	step = epoch * num_batch
	for i in range(num_batch):
		images, gt_texts, train_masks, kernal_images = pre_tools.batch_data(data_loader, i, FLAGS.batch_size)

		images = tf.convert_to_tensor(images, dtype=tf.float32)
		gt_texts = tf.convert_to_tensor(gt_texts, dtype=tf.float32)
		train_masks = tf.convert_to_tensor(train_masks, dtype=tf.float32)
		kernal_images = tf.convert_to_tensor(kernal_images, dtype=tf.float32)

		with tf.GradientTape() as tape:
			logites = model(images) ### (32,320,320,3)
			logites = tf.transpose(logites, (0, 3, 1, 2)) ### (32,3,320,320)
			Loss = PM.Dec_Loss_2(logites=logites, gt_texts=gt_texts,
								 gt_kernels=kernal_images, training_masks=train_masks,kernal=FLAGS.kernal_num)

		# 计算梯度 tape模式，保持跟踪
		grads = tape.gradient(Loss, model.trainable_weights)
		optim.apply_gradients(zip(grads, model.trainable_weights))

		texts = logites[:, 0, :, :]
		kernels = logites[:, 1:, :, :]
		score_text = cal_text_score(texts, gt_texts, train_masks, running_metric_text)
		score_kernel = cal_kernel_score(kernels, kernal_images, gt_texts, train_masks, running_metric_kernel)
		acc = score_text['Mean Acc']
		iou_t = score_text['Mean IoU']
		iou_k = score_kernel['Mean IoU']
		step = step + 1

		if i % 20 == 0:
			information = '## Epoch:{:d}  Step_Train / Total_Batch: {:d} / {:d}  train_loss= {:5f}  train_acc= {:5f} IOU_t={:5f} IOU_k={:5f}'. \
				format(epoch,step, num_batch, Loss, acc,iou_t,iou_k)
			print(information)  ### 输出到屏幕
			#logging.info(information)  ### 输出到log文件

def main():
	model_save_dir = "./checkpoints/"
	train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
	model_name = 'PSENet_BC-{:d}_k{:d}_{:s}.ckpt'.format(FLAGS.batch_size, FLAGS.kernal_num, str(train_start_time))
	model_save_path = os.path.join(model_save_dir, model_name)
	logging.basicConfig(filename='./checkpoints/' + model_name + '.log',
						format='%(asctime)s - %(pathname)s - %(levelname)s: %(message)s',
						level=logging.DEBUG, filemode='a', datefmt='%Y-%m-%d%I:%M:%S %p')

	model = PM.resnet18(num_classes=FLAGS.kernal_num)

	data_path = data_pre.get_data_path()
	data_num = len(data_path)
	num_batch = int(data_num / FLAGS.batch_size)

	for epoch in range(FLAGS.epochs):
		data_loader = data_pre.Data_load_pre(data_path, FLAGS.kernal_num)  ### random shuffle

		learning_rate = FLAGS.learning_rate
		if epoch in [200,400]:
			learning_rate = FLAGS.learning_rate *0.1
		train(data_loader, model, epoch,num_batch,learning_rate)
		model.save(model_save_path)



if __name__=="__main__":
	main()
