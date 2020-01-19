#! /usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
def conv2d(input,input_channel, out_channel, kernel_size, padding='SAME',stride=1, data_format='NHWC', name=None):
	with tf.variable_scope(name):
		w = tf.get_variable("weight", [kernel_size, kernel_size, input_channel, out_channel], initializer=tf.truncated_normal_initializer(stddev=0.1))
		b = tf.get_variable("biase", [out_channel], initializer=tf.truncated_normal_initializer(0.0))
		conv = tf.nn.conv2d(input, w,strides=[1, stride, stride, 1], padding=padding, data_format=data_format)
		ret = tf.identity(tf.nn.bias_add(conv, b, data_format=data_format),name=name)
	return ret

def unpool(inputs,name,fac=2):
	####  上采样
	with tf.variable_scope(name_or_scope=name):
		#if data_format=='channels_last' else tf.transpose(inputs,[0,2,3,1])
		upconv=tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*int(fac), tf.shape(inputs)[2]*int(fac)])
		###  是否加 BN 和 Relu
	return upconv

def unpool_2(inputs_1,inputs_2,name):
	####  上采样
	with tf.variable_scope(name_or_scope=name):
		#if data_format=='channels_last' else tf.transpose(inputs,[0,2,3,1])
		upconv=tf.image.resize_bilinear(inputs_1, size=[tf.shape(inputs_2)[1], tf.shape(inputs_2)[2]])
		###  是否加 BN 和 Relu
	return upconv

def _conv(input,in_dims,out_dims,kernel_size,stride,is_training,name):
	with tf.variable_scope(name_or_scope=name):
		conv = conv2d(input=input, input_channel=in_dims,out_channel=out_dims,kernel_size=kernel_size,
					  stride=stride, name='conv')
		bn = tf.layers.batch_normalization(inputs=conv,  name='bn',training=is_training)
		relu = tf.nn.relu(features=bn, name=name)
		pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	return pool
