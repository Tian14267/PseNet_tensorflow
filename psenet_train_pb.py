# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import time
import logging
import numpy as np
from model_net import psenet_model as PM
from model_net import model_v1
from data_pre import data_pre,pre_tools
from tensorflow.python.framework.graph_util import convert_variables_to_constants
os.environ['CUDA_VISIBLE_DEVICES']='0'
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

flages=tf.flags
FLAGS = flages.FLAGS
flages.DEFINE_float("learning_rate",0.0005,"learning rate")
flages.DEFINE_integer("kernal_num", 3, "kernal number")
flages.DEFINE_boolean("model_restore",False,"wheater to restore model")
flages.DEFINE_integer("batch_size", 8, "batch size")
flages.DEFINE_integer("epochs", 800, "epochs of net")


def train_PSE():
	data_path = data_pre.get_data_path()
	data_num = len(data_path)
	num_batch = int(data_num / FLAGS.batch_size)

	tf_image = tf.placeholder(dtype=tf.float32,shape=[FLAGS.batch_size,None,None,3],name="image")
	tf_gt = tf.placeholder(dtype=tf.float32,shape=[FLAGS.batch_size,None,None],name="gt_text")
	tf_mask = tf.placeholder(dtype=tf.float32,shape=[FLAGS.batch_size,None,None],name="mask")
	tf_kernal = tf.placeholder(dtype=tf.float32,shape=[FLAGS.batch_size,FLAGS.kernal_num - 1,None,None],name="kernal_image")

	resnet = PM.ResNet(PM.BottleBlock(),FLAGS.kernal_num,True,0.5)
	logites = resnet(tf_image) ## (batch, 7, size, size)

	#logites,_ = model_v1.model(tf_image, FLAGS.kernal_num)

	#Loss = PM.Dec_Loss(logites=logites,gt_texts=tf_gt,gt_kernels=tf_kernal,training_masks=tf_mask)
	Loss = PM.Dec_Loss_1(logites=logites, gt_text=tf_gt, gt_kernals=tf_kernal, train_mask=tf_mask)

	global_step = tf.Variable(0, name='global_step', trainable=False)

	
	learning_rate = tf.train.exponential_decay(
		learning_rate=FLAGS.learning_rate,
		global_step=global_step,
		decay_steps=num_batch * 200,
		decay_rate=0.5,
		staircase=True)
	'''
	boundaries = [200, 400]
	learing_rates = [0.001,0.0005,0.00025]
	learning_rate = tf.train.piecewise_constant(global_step, boundaries=boundaries, values=learing_rates)
	'''
	optim = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=Loss, global_step=global_step)
	#ema = tf.train.ExponentialMovingAverage(0.9999,global_step)
	#optim = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.99).minimize(loss=Loss, global_step=global_step)
	#optim = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=Loss, global_step=global_step)

	tensorboard_dir = "tensorboard"
	tf.summary.scalar("loss",Loss)
	merged_summary = tf.summary.merge_all()
	writer = tf.summary.FileWriter(tensorboard_dir)

	model_save_dir = "./checkpoints/pretrain_train/"
	train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
	model_name = 'PSENet_BC-{:d}_k{:d}_{:s}.ckpt'.format(FLAGS.batch_size,FLAGS.kernal_num,str(train_start_time))
	model_save_path = os.path.join(model_save_dir, model_name)
	logging.basicConfig(filename='./checkpoints/' + model_name + '.log',
						format='%(asctime)s - %(pathname)s - %(levelname)s: %(message)s',
						level=logging.DEBUG, filemode='a', datefmt='%Y-%m-%d%I:%M:%S %p')

	sess_config = tf.ConfigProto(allow_soft_placement=True)
	sess_config.gpu_options.allow_growth = True
	sess = tf.Session(config=sess_config)

	with sess.as_default():
		if FLAGS.model_restore:
			print("None!")
		else:
			step = 0
			init = tf.global_variables_initializer()
			sess.run(init)
			writer.add_graph(sess.graph)
		print("First step is:",step)

		Begain_learn_rate = FLAGS.learning_rate
		train_acc = 0
		start_epoch = 0
		if FLAGS.model_restore:
			start_epoch = int(step / num_batch)
		for epoch in range(start_epoch,FLAGS.epochs):
			data_loader = data_pre.Data_load_pre(data_path,FLAGS.kernal_num) ### random shuffle

			for i in range(num_batch):
				images, gt_texts, train_masks, kernal_images = pre_tools.batch_data(data_loader, i, FLAGS.batch_size)
				step += 1
				_,learn_rate,loss,train_pred,merge_summary_value = sess.run([optim,learning_rate,Loss,logites,merged_summary],
					feed_dict={tf_image:images,tf_gt:gt_texts,tf_mask:train_masks,tf_kernal:kernal_images})

				############# 输出 learning_rate  ###############################
				if Begain_learn_rate != learn_rate:
					information = "############ New Learning_Rate {:6f} in step {:d}  ###########".format(learn_rate,step)
					logging.info(information)
					print(information)
					Begain_learn_rate = learn_rate

				if step % 20 == 0:
					score_text = pre_tools.calcult_acc(train_pred[:,0,:,:],gt_texts,train_masks,pre_tools.runningScore(2))
					acc = score_text['Mean Acc']
					information = '## Epoch {:d} Step_Train / Total_Batch: {:d} / {:d}  train_loss= {:5f}  train_acc= {:5f}'. \
						format(epoch, step, num_batch, loss,acc)
					print(information)  ### 输出到屏幕
					logging.info(information)  ### 输出到log文件
				
				#if step % (num_batch * 1) == 0:  ### 每 35个epoch进行一次验证，并保存最优模型
				if step % 100 == 0:  ### 每 35个epoch进行一次验证，并保存最优模型
					#if train_acc <  acc:
					print("############# Save model in Epoch {:d} Step_Train / Total_Batch {:d} / {:d} ############".\
						  								format(epoch,step,num_batch))
					#saver.save(sess, model_save_path, global_step=step)
					graph_def = tf.get_default_graph().as_graph_def() 
					output_graph_def = convert_variables_to_constants(sess, sess.graph_def, output_node_names=['transpose'])
					model_f = tf.gfile.GFile("./checkpoints/pb/pse.pb","wb")
					model_f.write(output_graph_def.SerializeToString())
					#train_acc = acc
			



if __name__=="__main__":
	train_PSE()
