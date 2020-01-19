#! /usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from model_net.bascalNet import _conv,unpool,unpool_2,conv2d

class BasicBlock(object):
	def __init__(self):
		return
	def __call__(self, input,is_training,name):
		result = self.forward(input,is_training,name)
		return result

	def forward(self, input,is_training,name):
		with tf.variable_scope(name):
			input_size = input.get_shape().as_list()
			conv1 = conv2d(input=input, input_channel=input_size[-1], out_channel=input_size[-1],
						   kernel_size=3, stride=1,name="conv1")
			bn1 = tf.layers.batch_normalization(inputs=conv1, training=is_training)
			relu1 = tf.nn.relu(features=bn1)

			conv2 = conv2d(input=relu1, input_channel=input_size[-1], out_channel=input_size[-1],
						   kernel_size=3, stride=1,name="conv2")
			bn = tf.layers.batch_normalization(inputs=conv2, training=is_training)

			out = bn + input
			relu = tf.nn.relu(features=out)
		return relu


class BottleBlock(object):### 残差块：1*3*1
	def __init__(self):
		return
	def __call__(self, input,is_training,name):
		result = self.forward(input,is_training,name)
		return result

	def forward(self,input,is_training,name):
		input_size = input.get_shape().as_list()
		with tf.variable_scope(name):
			conv1 = conv2d(input=input, input_channel=input_size[-1], out_channel=int(input_size[-1] / 4),
						   kernel_size=1,stride=1,name = "conv_1")
			bn1 = tf.layers.batch_normalization(inputs=conv1,training=is_training)
			relu1 = tf.nn.relu(features=bn1)

			conv2 = conv2d(input=relu1, input_channel=int(input_size[-1] / 4), out_channel=int(input_size[-1] / 4),
						   kernel_size=3, stride=1,name = "conv_2")
			bn2 = tf.layers.batch_normalization(inputs=conv2,training=is_training)
			relu2 = tf.nn.relu(features=bn2)

			conv3 = conv2d(input=relu2, input_channel=int(input_size[-1] / 4), out_channel=input_size[-1],
						   kernel_size=1, stride=1,name = "conv_3")
			bn3 = tf.layers.batch_normalization(inputs=conv3,training=is_training)
			relu3 = tf.nn.relu(features=bn3)

			out = relu3 + input
			relu = tf.nn.relu(features=out)
			return relu


class ResNet(object):
	def __init__(self,block,kernal_num,is_training,keep_drop): ### block: BasicBlock()  or  BottleBlock()
		#self._is_training = is_training
		self.kernal_num = kernal_num
		self.block = block
		self.is_training = is_training
		self.keep_drop = keep_drop

	def __call__(self, input):
		result = self.forward(input)
		return result

	def _make_layer(self,input,out_dims,training,name):
		input_size = input.get_shape().as_list()
		with tf.variable_scope(name_or_scope=name):
			conv_out = _conv(input, input_size[-1], out_dims, 3, 1, training, "make_conv_1")
			conv = conv2d(input=conv_out, input_channel=out_dims, out_channel=out_dims, kernel_size=3,stride=1, name='make_conv_2')
			bn = tf.layers.batch_normalization(inputs=conv, training=training,name='make_bn')
			relu = tf.nn.relu(features=bn)
			return relu

	def _up_conv_concat(self,Block1,Block2,Block3,Block4):
		# Block4
		conv_B4 = conv2d(input=Block4, input_channel=Block4.get_shape().as_list()[-1], out_channel=64, kernel_size=2,
						 stride=1, name='conv_B4')

		uplayer_B4 = unpool_2(conv_B4,Block3, "up_layer_B4")
		uplayer_B4_out = unpool_2(conv_B4,Block1, "up_layer_B4_2")  ## (8,256,256,64)

		## Block3
		conv_B3 = conv2d(input=Block3, input_channel=Block3.get_shape().as_list()[-1], out_channel=64, kernel_size=2,
						 stride=1, name='conv_B3')
		Sum_3 = conv_B3 + uplayer_B4
		uplayer_B3 = unpool_2(Sum_3, Block2,"up_layer_B3")
		uplayer_B3_out = unpool_2(Sum_3,Block1, "up_layer_B3_2")  ## (8,256,256,64)

		### Block2
		conv_B2 = conv2d(input=Block2, input_channel=Block2.get_shape().as_list()[-1], out_channel=64, kernel_size=2,
						 stride=1, name='conv_B2')
		Sum_2 = conv_B2 + uplayer_B3
		uplayer_B2 = unpool_2(Sum_2, Block1,"up_layer_B2")  ## (8,256,256,64)

		#### Block1
		conv_B1 = conv2d(input=Block1, input_channel=Block1.get_shape().as_list()[-1], out_channel=64, kernel_size=2,
						 stride=1, name='conv_B1')
		Sum_1 = conv_B1 + uplayer_B2  ## (8,256,256,64)

		concat_out = tf.concat([uplayer_B4_out, uplayer_B3_out, uplayer_B2, Sum_1], axis=-1)

		return concat_out

	def _cycle_block(self,input,block_num,name):
		block_value = input
		with tf.variable_scope(name):
			block_contant = []
			for i in range(block_num):
				block_value = self.block(block_value,self.is_training, "Block_%d"%(i))
				block_contant.append(block_value)
		return block_value,block_contant

	def forward(self,input):
		input_size = input.get_shape().as_list()  ### (batch,1024,1024,3)

		#####  CNN forword Blocks  ###################################################
		conv_out = _conv(input,input_size[-1],64, 7, 2,self.is_training, "conv1") ## (8,256,256,64)

		Block1,Blocks_1 = self._cycle_block(conv_out,3,"Block_cycle_1") ## (8,256,256,64)
		layer_1 = self._make_layer(Block1, 128,self.is_training, "make_layer_1") ## (8,128,128,128)

		Block2,Blocks_2 = self._cycle_block(layer_1, 4, "Block_cycle_2")  ## (8,128,128,128)
		layer_2 = self._make_layer(Block2, 256,self.is_training, "make_layer_2") ## (8,64,64,256)

		Block3,Blocks_3 = self._cycle_block(layer_2, 6, "Block_cycle_3")  ## (8,64,64,256)
		layer_3 = self._make_layer(Block3, 512,self.is_training, "make_layer_3") ## (8,32,32,512)

		Block4,Blocks_4 = self._cycle_block(layer_3, 3, "Block_cycle_4")  ## (8,32,32,512)
		####### 上采样 Concat ########################################################
		Concat = self._up_conv_concat(Block1,Block2,Block3,Block4)  ### (8,256,256,256)

		conv_end = conv2d(input=Concat, input_channel=Concat.get_shape().as_list()[-1], out_channel=self.kernal_num,
							kernel_size=2,stride=1, name='conv_end')
		bn_end = tf.layers.batch_normalization(inputs=conv_end, name='bn_end',momentum=0.999, epsilon=1e-3)
		relu_end = tf.nn.relu(features=bn_end, name="relu_end")  ### (8,256,256,7)

		dropout = tf.nn.dropout(relu_end, self.keep_drop)

		out = unpool_2(dropout, input,"ResNet_Out") ### (8, 1024, 1024, 7)
		out = tf.transpose(out,[0,3,1,2]) ## (8, 7, 1024, 1024)

		return out

###############################################################
###############################################################
####            Loss Calculate  tensorflow
def OHEM( gt, pred):
	# online hard example miniing: 在线难例挖掘
	# NOTE: OHM 3
	# input is a tensor
	# (batch, h, w)
	batch = gt.get_shape().as_list()[0]
	pred_new = []
	for i in range(batch):
		gt_oneBatch = gt[i,:,:]
		pred_oneBatch = pred[i,:,:]
		pos_mask = tf.cast(tf.equal(gt_oneBatch, 1.), dtype=tf.float32)  # [h,w,1]
		neg_mask = tf.cast(tf.equal(gt_oneBatch, 0.), dtype=tf.float32)
		n_pos = tf.reduce_sum((pos_mask), [0, 1]) ## [0, 1] 将两个维度加在一起，得到总和

		neg_val_all = tf.boolean_mask(pred_oneBatch, neg_mask)  # [N]
		n_neg = tf.minimum(tf.shape(neg_val_all)[-1], tf.cast(n_pos * 3, tf.int32))
		#print("Shape:",n_neg.shape)
		#n_neg = tf.cond(tf.greater(n_pos, 0), lambda: n_neg, lambda: tf.shape(neg_val_all)[-1])
		neg_hard, neg_idxs = tf.nn.top_k(neg_val_all, k=n_neg)  # [batch_size,k][batch_size, k]
		# TODO ERROR  slice index -1 of dimension 0 out of bounds.
		neg_min = tf.cond(tf.greater(tf.shape(neg_hard)[-1], 0), lambda: neg_hard[-1], lambda: 1.)  # [k]

		neg_hard_mask = tf.cast(tf.greater_equal(pred_oneBatch, neg_min), dtype=tf.float32)
		pred_ohm = pos_mask * pred_oneBatch + neg_hard_mask * neg_mask * pred_oneBatch
		pred_new.append(pred_ohm)
	pred_new = tf.stack(pred_new,axis=0)

	return pred_new, gt

def calculate_dice_loss(pred, gt):
	###   Lc loss
	# pred: (batch，h, w)
	# gt: (batch，h,w)
	union = tf.reduce_sum(tf.multiply(pred, gt),[1,2])  ## 乘法：相同位置的元素相乘
	pred_union = tf.reduce_sum(tf.square(pred),[1,2])
	gt_union = tf.reduce_sum(tf.square(gt), [1, 2])

	dice_loss = 1. - (2 * union + 1e-5) / (pred_union + gt_union + 1e-5)

	return dice_loss

def Dec_Loss(logites,gt_texts,gt_kernels,training_masks):
	'''
	 L = λLc + (1 − λ)Ls
	:param logites: shape: (batch, kernal_num, h, w)
	:param gt_texts: shape: (batch, h, w)
	:param gt_kernels:shape: (batch, kernal_num - 1, h, w)
	:param training_masks:shape: (batch, h, w)
	:return: Loss
	'''
	with tf.name_scope("Loss"):
		pred_text = logites[:,0,:,:]
		pred_kernals = logites[:,1:,:,:]

		pred_text_map = pred_text * training_masks ## (batch, h, w)
		gt_text_map = gt_texts * training_masks  ## (batch, h, w)

		pred_text_maps, gt_text_maps = OHEM(gt_text_map, pred_text_map)

		lc_loss = calculate_dice_loss(pred_text_maps,gt_text_maps)  ### (batch, Lc_loss)

		num = pred_kernals.get_shape().as_list()[1]
		mask = tf.to_float(tf.greater(pred_text * training_masks, 0.5))
		#mask = tf.to_float(tf.where((pred_text * train_mask)>=0.5,1,0))
		Ls_loss = 0
		for i in range(num):
			pred_kernal = pred_kernals[:,i,:,:]
			gt_kernal = gt_kernels[:,i,:,:]

			pred_kernal = pred_kernal * mask
			gt_kernal = gt_kernal * mask

			ls_loss = calculate_dice_loss(pred_kernal,gt_kernal) ## (batch,)
			Ls_loss = Ls_loss + tf.reduce_mean(ls_loss)

		Lc_Loss = 0.7 * tf.reduce_mean(lc_loss)
		Ls_Loss = 0.3 * Ls_loss / num

	return (Lc_Loss + Ls_Loss)
###############################################################

