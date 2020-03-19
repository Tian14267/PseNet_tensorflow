# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import time
import logging

import random
from model_net import psenet_model_2 as PM
from data_pre import data_pre,pre_tools
os.environ['CUDA_VISIBLE_DEVICES']='0'

flages=tf.flags
FLAGS = flages.FLAGS
flages.DEFINE_float("learning_rate",0.0005,"learning rate")
flages.DEFINE_integer("kernal_num", 3, "kernal number")
flages.DEFINE_boolean("model_restore",False,"wheater to restore model")
flages.DEFINE_boolean("model_pretain",True,"")
flages.DEFINE_integer("batch_size", 8, "batch size")
flages.DEFINE_integer("epochs", 10, "epochs of net")
#################################
#####   注意：
#####   1：batch要与预训练的batch一致；2：所有非sess的内容，必须在Graph中

def train_PSE():
	data_path = data_pre.get_data_path()
	data_num = len(data_path)
	num_batch = int(data_num / FLAGS.batch_size)

	with tf.Graph().as_default() as g:
		saver = tf.train.import_meta_graph('./checkpoints/pretrain_train/PSENet_BC-8_k3_2020-03-19-15-57-47.ckpt-6000.meta')
		tf_image = g.get_tensor_by_name('image:0')
		tf_gt = g.get_tensor_by_name('gt_text:0')
		tf_mask = g.get_tensor_by_name('mask:0')
		tf_kernal = g.get_tensor_by_name('kernal_image:0')

		Block1 = g.get_tensor_by_name('Block_cycle_1/Block_2/block_relu_out:0')
		Block2 = g.get_tensor_by_name('Block_cycle_2/Block_3/block_relu_out:0')
		Block3 = g.get_tensor_by_name('Block_cycle_3/Block_5/block_relu_out:0')
		Block4 = g.get_tensor_by_name('Block_cycle_4/Block_2/block_relu_out:0')

		Block1 = tf.stop_gradient(Block1, name='stop_gradient_1')
		Block2 = tf.stop_gradient(Block2, name='stop_gradient_2')
		Block3 = tf.stop_gradient(Block3, name='stop_gradient_3')
		Block4 = tf.stop_gradient(Block4, name='stop_gradient_4')

		logites = PM.finetuning_inference(tf_image,Block1,Block2,Block3,Block4,FLAGS.kernal_num)
		Loss = PM.Dec_Loss_1(logites=logites, gt_text=tf_gt, gt_kernals=tf_kernal, train_mask=tf_mask)
		global_step = tf.Variable(0, name='global_step', trainable=False)

		learning_rate = tf.train.exponential_decay(
            learning_rate=FLAGS.learning_rate,
            global_step=global_step,
            decay_steps=1000,
            decay_rate=0.5,
            staircase=True)
		optim = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=Loss, global_step=global_step)

		tensorboard_dir = "tensorboard"
		tf.summary.scalar("loss",Loss)
		merged_summary = tf.summary.merge_all()
		writer = tf.summary.FileWriter(tensorboard_dir)

		#saver = tf.train.Saver(max_to_keep=3)  ### 保存模型
		model_save_dir = "checkpoints/model/"
		train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
		model_name = 'PSENet_{:s}.ckpt'.format(str(train_start_time))
		model_save_path = os.path.join(model_save_dir, model_name)
		logging.basicConfig(filename='./checkpoints/model/' + model_name + '.log',
                            format='%(asctime)s - %(pathname)s - %(levelname)s: %(message)s',
                            level=logging.DEBUG, filemode='a', datefmt='%Y-%m-%d%I:%M:%S %p')

    ####  以上代码必须全部放在Graph里面
    #sess_config = tf.ConfigProto(allow_soft_placement=True)
    #sess_config.gpu_options.allow_growth = True
    #sess = tf.Session(config=sess_config)

	with tf.Session(graph=g) as sess:
		if FLAGS.model_pretain:
			saver.restore(sess=sess, save_path='./checkpoints/pretrain_train/PSENet_BC-8_k3_2020-03-19-15-57-47.ckpt-6000')
			sess.run(tf.global_variables_initializer())
		if FLAGS.model_restore:
			weights_path = ""
			saver.restore(sess = sess,save_path=weights_path)
			step = sess.run(tf.train.get_global_step())
			writer.add_graph(sess.graph)
			infor = '##### Restore model : ' + weights_path + '  ########'
			logging.info(infor)
			print(infor)
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
				#images, gt_texts, train_masks, kernal_images = pre_tools.batch_data(data_loader,data_array, i, FLAGS.batch_size)
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

				if step % 10 == 0:
					score_text = pre_tools.calcult_acc(train_pred[:,0,:,:],gt_texts,train_masks,pre_tools.runningScore(2))
					acc = score_text['Mean Acc']
					information = '## Epoch {:d} Step_Train / Total_Batch: {:d} / {:d}  train_loss= {:5f}  train_acc= {:5f}'. \
						format(epoch, step, num_batch, loss,acc)
					print(information)  ### 输出到屏幕
					logging.info(information)  ### 输出到log文件
					if step % 500 == 0:  ### 每 500 步进行一次验证，并保存最优模型
						if train_acc <  acc:
							print("#############  Save model in Epoch {:d} Step_Train / Total_Batch {:d} / {:d}  ############".\
							  								format(epoch,step,num_batch))
							saver.save(sess, model_save_path, global_step=step)
							train_acc = acc


if __name__=="__main__":
    train_PSE()
