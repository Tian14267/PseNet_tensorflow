import model_net.net_v1.resnet_model as resnet_model
from model_net.net_v1.resnet_model import batch_norm

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers

def _get_block_sizes(resnet_size):
	"""Retrieve the size of each block_layer in the ResNet model.

	The number of block layers used for the Resnet model varies according
	to the size of the model. This helper grabs the layer set we want, throwing
	an error if a non-standard size has been selected.

	Args:
	resnet_size: The number of convolutional layers needed in the model.

	Returns:
	A list of block sizes to use in building the model.

	Raises:
	KeyError: if invalid resnet_size is received.
	"""
	choices = {
	  18: [2, 2, 2, 2],
	  34: [3, 4, 6, 3],
	  50: [3, 4, 6, 3],
	  101: [3, 4, 23, 3],
	  152: [3, 8, 36, 3],
	  200: [3, 24, 36, 3]
	}

	try:
		return choices[resnet_size]
	except KeyError:
		err = ('Could not find layers for selected Resnet size.\n'
			'Size received: {}; sizes allowed: {}.'.format(
			resnet_size, choices.keys()))
		raise ValueError(err)


class ImagenetModel(resnet_model.Model):
	"""Model class with appropriate defaults for Imagenet data."""

	def __init__(self, resnet_size, data_format=None, num_classes=None,
				 resnet_version=resnet_model.DEFAULT_VERSION,
				 dtype=resnet_model.DEFAULT_DTYPE):
		"""These are the parameters that work for Imagenet data.

		Args:
		  resnet_size: The number of convolutional layers needed in the model.
		  data_format: Either 'channels_first' or 'channels_last', specifying which
			data format to use when setting up the model.
		  num_classes: The number of output classes needed from the model. This
			enables users to extend the same model to their own datasets.
		  resnet_version: Integer representing which version of the ResNet network
			to use. See README for details. Valid values: [1, 2]
		  dtype: The TensorFlow dtype to use for calculations.
		"""

		# For bigger models, we want to use "bottleneck" layers
		if resnet_size < 50:
			bottleneck = False
		else:
			bottleneck = True

		super(ImagenetModel, self).__init__(
			resnet_size=resnet_size,
			bottleneck=bottleneck,
			num_classes=num_classes,
			num_filters=64,
			kernel_size=7,
			conv_stride=2,
			first_pool_size=3,
			first_pool_stride=2,
			block_sizes=_get_block_sizes(resnet_size),
			block_strides=[1, 2, 2, 2],
			resnet_version=resnet_version,
			data_format=data_format,
			dtype=dtype
		)

def unpool(inputs, fac=2,data_format='channels_last'):
	inputs=inputs if data_format=='channels_last' else tf.transpose(inputs,[0,2,3,1])
	inputs=tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*int(fac),  tf.shape(inputs)[2]*int(fac)])
	return inputs if data_format=='channels_last' else tf.transpose(inputs,[0,3,1,2])

def conv_layer(input,output_dep,filter_size,is_training=True,data_format='channels_first'):
	input=tf.layers.conv2d(input,256,1,kernel_initializer=initializers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False),data_format=data_format)
	input=batch_norm(input,training=is_training,data_format=data_format)
	input=tf.nn.relu(input)
	return input

def model(inputs,kernal_num,data_format='channels_last',is_training=True):
	## inputs : [batch, ?,? 3]
	#inputs = tf.transpose(inputs,[0,3,1,2])
	model_class = ImagenetModel(
		resnet_size=50, data_format=data_format,num_classes=0,resnet_version=1)
	_, end_points = model_class(inputs, is_training)

	f = [end_points['block_layer4'], end_points['block_layer3'],
		 end_points['block_layer2'], end_points['block_layer1']]
	P = [None, None, None, None]
	# attent=[None,None,None,None]
	# import pudb; pudb.set_trace()
	with tf.variable_scope('fpn_module'):
		for i in range(4):
			if i == 0:
				# TODO: I don't sure whether this nedd bn
				P[i]=tf.layers.conv2d(f[i],256,1,data_format=data_format)
				P[i]=batch_norm(P[i],training=is_training,data_format=data_format)
				P[i]=tf.nn.relu(P[i])

				F_multi = unpool(P[i], 8,data_format=data_format)
			else:
				P[i] = tf.layers.conv2d(f[i], 256, 1,data_format=data_format)
				P[i]=batch_norm(P[i],training=is_training,data_format=data_format)
				P[i]=tf.nn.relu(P[i])

				# attent[i]=slim.conv2d(P[i],1,1)
				# P[i]=P[i]*attent[i]

				P[i] = unpool(P[i-1],data_format=data_format)+P[i]
				F_multi = tf.concat((F_multi, unpool(
					P[i], 8/(2**i), data_format=data_format)), 1 if data_format == 'channels_first' else -1)

		# with tf.variable_scope('deformable'):

		# fusion feature
		F_multi=tf.layers.conv2d(F_multi,256,3,padding='same',data_format=data_format)
		F_multi=batch_norm(F_multi,training=is_training,data_format=data_format)
		feat=tf.nn.relu(F_multi)

		for i in range(kernal_num):
			seg_map=tf.layers.conv2d(feat,1,1,data_format=data_format)
			seg_map = tf.sigmoid(unpool(seg_map, 4,data_format=data_format))
			if i == 0:
				seg_maps = seg_map
			else:
				seg_maps = tf.concat((seg_maps, seg_map), 1 if data_format == 'channels_first' else -1)

		if is_training==False:
			with tf.variable_scope('format_change'):
				seg_maps=tf.transpose(seg_maps,[0,2,3,1])  ## [batch, ?, ?, 3]
		#print("### Seg_maps Shapes of output of model:",seg_maps.shape)
		seg_maps = tf.transpose(seg_maps,[0,3,1,2]) ## [batch, ?, ?, 3]==>## [batch, 3, ?, ?]
		return seg_maps, f


if __name__=="__main__":
	input = tf.placeholder(tf.float32, [8, 2240, 2240, 3])
	is_training = 'train'
	output = model(input,3)
	print("Down!")
