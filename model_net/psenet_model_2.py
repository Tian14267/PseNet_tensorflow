#! /usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from model_net.bascalNet import _conv,unpool_2,conv2d

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
			relu = tf.nn.relu(features=out,name = 'block_relu_out')
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
			relu = tf.nn.relu(features=out,name='block_relu_out')
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

		pool_out = unpool_2(dropout, input,"ResNet_Out") ### (8, 1024, 1024, 7)
		out = tf.transpose(pool_out, [0, 3, 1, 2],name = 'transpose') ## (8, 7, 256, 256)
		

		return out


def finetuning_inference(input,block1,block2,block3,block4,kernal_num):
	####  说明：进行finetuning使用。在最后多加一层卷积
	#
	resnet = ResNet(BottleBlock(), kernal_num, True, 0.5)
	Concat = resnet._up_conv_concat(block1, block2, block3, block4)
	conv_2 = conv2d(input=Concat, input_channel=Concat.get_shape().as_list()[-1], out_channel=64,
					  kernel_size=2, stride=1, name='conv2')
	bn_2 = tf.layers.batch_normalization(inputs=conv_2, name='bn_2', momentum=0.999, epsilon=1e-3)
	relu_2 = tf.nn.relu(features=bn_2, name="relu_2")  ### (8,256,256,64)

	conv_3 = conv2d(input=relu_2, input_channel=relu_2.get_shape().as_list()[-1], out_channel=kernal_num,
					kernel_size=2, stride=1, name='conv3')
	bn_3 = tf.layers.batch_normalization(inputs=conv_3, name='bn_3', momentum=0.999, epsilon=1e-3)
	relu_3 = tf.nn.relu(features=bn_3, name="relu_3")  ### (8,256,256,7)

	dropout = tf.nn.dropout(relu_3, resnet.keep_drop)
	pool_out = unpool_2(dropout, input,"ResNet_Out") ### (8, 256, 256, 7)
	transpose = tf.transpose(pool_out, [0, 3, 1, 2], name='transpose')  ## (8, 7, 256, 256)

	return transpose

###############################################################
###############################################################
####            Loss Calculate  tensorflow
def OHEM( gt, pred):
	# online hard example miniing: 在线难例挖掘
	# NOTE: OHM 3
	# (batch, h, w)
	batch = gt.get_shape().as_list()[0]
	pred_new = []
	for i in range(batch):
		gt_oneBatch = gt[i,:,:]
		pred_oneBatch = pred[i,:,:]
		pos_mask = tf.cast(tf.equal(gt_oneBatch, 1.), dtype=tf.float32)  # [h,w,1]
		neg_mask = tf.cast(tf.equal(gt_oneBatch, 0.), dtype=tf.float32)
		n_pos = tf.reduce_sum((pos_mask), [0, 1])

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

def Dec_Loss_1(logites,gt_text,gt_kernals,train_mask):
	'''
	 L = λLc + (1 − λ)Ls
	:param logites: shape: (batch, kernal_num, h, w)
	:param gt_text: shape: (batch, h, w)
	:param gt_kernals:shape: (batch, kernal_num - 1, h, w)
	:param train_mask:shape: (batch, h, w)  标记了文本不清晰位置，即反例。反例=0，其余=1
	:return: Loss
	'''
	with tf.name_scope("Loss"):
		pred_text = logites[:,0,:,:]
		pred_kernals = logites[:,1:,:,:]

		pred_text_map = pred_text * train_mask ## (batch, h, w)
		gt_text_map = gt_text * train_mask  ## (batch, h, w)

		pred_text_maps, gt_text_maps = OHEM(gt_text_map, pred_text_map)

		lc_loss = calculate_dice_loss(pred_text_maps,gt_text_maps)  ### (batch, Lc_loss)

		num = pred_kernals.get_shape().as_list()[1]
		mask = tf.to_float(tf.greater(pred_text * train_mask, 0.5))
		#mask = tf.to_float(tf.where((pred_text * train_mask)>=0.5,1,0))
		Ls_loss = 0
		for i in range(num):
			pred_kernal = pred_kernals[:,i,:,:]
			gt_kernal = gt_kernals[:,i,:,:]

			pred_kernal = pred_kernal * mask
			gt_kernal = gt_kernal * mask

			ls_loss = calculate_dice_loss(pred_kernal,gt_kernal) ## (batch,)
			Ls_loss = Ls_loss + tf.reduce_mean(ls_loss)

		Lc_Loss = 0.7 * tf.reduce_mean(lc_loss)
		Ls_Loss = 0.3 * Ls_loss / num

	return (Lc_Loss + Ls_Loss)
###############################################################
##  torch --> tensorflow
def ohem_batch(scores, gt_texts, training_masks):
	####  tensorflow 版本
	'''
	:param scores: (batch, h, w)  Tensor
	:param gt_texts: (batch, h, w)  Tensor
	:param training_masks:(batch, h, w)  Tensor
	:return:
	'''
	selected_masks = []
	batch = scores.get_shape().as_list()[0]
	for i in range(batch):
		one_selcet_mask = ohem_single(scores[i, :, :], gt_texts[i, :, :], training_masks[i, :, :])
		selected_masks.append(one_selcet_mask)
	selected_masks = tf.concat(selected_masks,axis=0)############
	return selected_masks

def ohem_single(score, gt_text, training_mask):
	'''
	获取一个 batch 的掩码数据
	:param score: (h, w) Tensor
	:param gt_text: (h, w) Tensor
	:param training_mask: (h, w) Tensor
	:return:
	'''
	#shape = gt_text.get_shape().as_list()  ## (h，w)
	shape = tf.shape(gt_text)
	gt_text_greater = tf.cast(tf.greater(gt_text, 0.5), tf.int32)  ## gt_text > 0.5
	gt_text_less = tf.cast(tf.greater_equal(0.5, gt_text), tf.int32)  ##  gt_text <= 0.5  (h, w)

	pos_score = tf.reduce_sum(gt_text_greater)
	equal_temp = tf.multiply(gt_text_greater, tf.cast(tf.greater_equal(0.5, training_mask),tf.int32)) ### (h, w)
	pos_num = pos_score - tf.reduce_sum(tf.cast(equal_temp, tf.int32))

	if pos_num == 0:
		selected_mask = tf.expand_dims(training_mask,axis=0) ## (1, h, w)
		selected_mask = tf.cast(selected_mask,tf.float32)
		return selected_mask

	neg_num = tf.reduce_sum(gt_text_less)### 非文本区域像素个数  gt_text <= 0.5
	neg_num = tf.cond(tf.greater(pos_num * 3, neg_num), lambda: pos_num * 3, lambda: neg_num)### 设置反例比例数量

	if neg_num == 0:
		selected_mask = tf.expand_dims(training_mask, axis=0)  ## (1, h, w)
		selected_mask = tf.cast(selected_mask, tf.float32)
		return selected_mask

	score_multiply = tf.multiply(score, tf.cast(gt_text_less,tf.float32)) ### 矩阵对应元素相乘
	neg_score = tf.reshape((-1.0) * score_multiply, shape=(shape[0] * shape[1],))  ###  (h*w, )
	neg_score_sorted = tf.sort(neg_score, direction='ASCENDING')
	threshold = -neg_score_sorted[neg_num - 1]

	pred_score = tf.logical_or(tf.greater(score, threshold),tf.greater(gt_text, 0.5))  ## 输入 bool 型：(h, w)，返回bool
	selected_mask = tf.logical_and(pred_score,tf.greater(training_mask, 0.5))  ## bool 型
	selected_mask = tf.expand_dims(tf.cast(selected_mask,tf.float32),axis=0)

	return selected_mask

def dice_loss(input, target, mask):
	'''
	计算 loss
	:param input: pred_texts: (batch, h, w)
	:param target: gt_texts:  (batch, h, w)
	:param mask:  selected_masks OHEM 返回的新 mask: (batch, h, w)
	:return:
	'''
	input = tf.sigmoid(input)
	#input_shape = input.get_shape().as_list()  ## (h，w)
	input_shape = tf.shape(input)
	input = tf.reshape(input,(input_shape[0], input_shape[1] * input_shape[2]))
	target = tf.reshape(target,(input_shape[0], input_shape[1] * input_shape[2]))
	mask = tf.reshape(mask, (input_shape[0], input_shape[1] * input_shape[2]))

	input = input * mask  ## ([4,102400]) 对应元素相乘
	target = target * mask  ## ([4,102400])对应元素相乘

	a = tf.reduce_sum(input * target,axis=1)
	b = tf.reduce_sum(input * input,axis=1)+ 0.001
	c = tf.reduce_sum(target * target, axis=1) + 0.001  # (4)
	d = (2 * a) / (b + c)  # (4)
	dice_loss = tf.reduce_mean(d)
	return 1 - dice_loss

def Dec_Loss(logites,gt_texts, gt_kernels,training_masks):
	'''
	 L = λLc + (1 − λ)Ls
	:param logites: (batch, kernal_num, h, w)
	:param gt_texts: (batch, h, w)
	:param gt_kernels: (batch, kernal_num - 1, h, w)
	:param training_masks: (batch, h, w)  标记了文本不清晰位置，即反例。反例=0，其余=1
	:return:
	'''
	with tf.name_scope("Loss"):
		texts = logites[:, 0, :, :]
		kernals = logites[:, 1:, :, :]
		kernals_shape = kernals.get_shape().as_list()

		selected_masks = ohem_batch(texts, gt_texts, training_masks)

		loss_text = dice_loss(texts, gt_texts, selected_masks)
		loss_kernels = []
		mask0 = tf.sigmoid(texts)
		mask1 = training_masks

		selected_masks = tf.logical_and(tf.greater(mask0, 0.5),tf.greater(mask1, 0.5))
		selected_masks = tf.cast(selected_masks,tf.float32)

		for i in range(kernals_shape[1]):
			kernel_i = kernals[:, i, :, :]
			gt_kernel_i = gt_kernels[:, i, :, :]
			loss_kernel_i = dice_loss(kernel_i, gt_kernel_i, selected_masks)
			loss_kernels.append(loss_kernel_i)
		loss_kernel = tf.reduce_sum(loss_kernels) / kernals_shape[1]
		loss = 0.7 * loss_text + 0.3 * loss_kernel

	return loss

#
###############################################################
#####   loss  numpy版本
"""
def ohem_single(score, gt_text, training_mask):
	pos_num = (int)(np.sum(gt_text > 0.5)) - (int)(np.sum((gt_text > 0.5) & (training_mask <= 0.5)))
	###  pos = gt中大于0.5个数  -   (gt中大于0.5个数 & mask中小于0.5)     #### 计算正例像素个数
	if pos_num == 0:
		# selected_mask = gt_text.copy() * 0 # may be not good
		selected_mask = training_mask
		selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
		return selected_mask

	neg_num = (int)(np.sum(gt_text <= 0.5))  ### 非文本区域像素个数
	neg_num = (int)(min(pos_num * 3, neg_num))  ### 设置反例比例数量

	if neg_num == 0:
		selected_mask = training_mask
		selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
		return selected_mask

	neg_score = score[gt_text <= 0.5]
	neg_score_sorted = np.sort(-neg_score)
	threshold = -neg_score_sorted[neg_num - 1]

	selected_mask = ((score >= threshold) | (gt_text > 0.5)) & (training_mask > 0.5)
	selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
	return selected_mask


def ohem_batch(scores, gt_texts, training_masks):
	#scores = scores.data.cpu().numpy()  ### Variable  转换成numpy
	#gt_texts = gt_texts.data.cpu().numpy()  ### Variable  转换成numpy
	#training_masks = training_masks.data.cpu().numpy()  ### 转换成numpy

	selected_masks = []
	for i in range(scores.shape[0]):  ### 通过正负比例 新建掩码
		selected_masks.append(ohem_single(scores[i, :, :], gt_texts[i, :, :], training_masks[i, :, :]))

	selected_masks = np.concatenate(selected_masks, 0)  ###### numpy  Concat
	#selected_masks = torch.from_numpy(selected_masks).float()  ## 转换成 tensor

	return selected_masks


def dice_loss(input, target, mask):  ### Lc loss
	def sigmod(x):
		return 1 / (1 + np.exp(-x))

	input = sigmod(input) ## (4,320,320)
	input = input.reshape((input.shape[0],input.shape[1]*input.shape[2]))

	target = target.reshape((target.shape[0], target.shape[1] * target.shape[2]))
	mask = mask.reshape((mask.shape[0], mask.shape[1] * mask.shape[2]))

	#input = input.contiguous().view(input.size()[0], -1)## (4,102400)
	#target = target.contiguous().view(target.size()[0], -1)## (4,102400)
	#mask = mask.contiguous().view(mask.size()[0], -1)## (4,102400)

	input = input * mask ## ([4,102400])
	target = target * mask ## ([4,102400])

	a = np.sum(input * target, axis=1)  # (4) ## input * target = (4,102400) , 第一维度相加
	b = np.sum(input * input, axis=1) + 0.001 # (4)
	c = np.sum(target * target, axis=1) + 0.001 # (4)
	d = (2 * a) / (b + c) # (4)
	dice_loss = np.mean(d)
	return 1 - dice_loss
"""

if __name__=="__main__":
	input = tf.placeholder(tf.float32, [8, 512, 512, 3])
	is_training = 'train'
	resnet = ResNet(BottleBlock(),7,True,0.5)
	out = resnet(input)
	print("Down!")
