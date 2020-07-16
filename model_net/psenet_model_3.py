import tensorflow as tf
import numpy as np
import sys

#sys.path.append('../')

def conv3x3(out_planes, strides=1):
	"""3x3 convolution with padding"""
	return tf.keras.layers.Conv2D(out_planes, kernel_size=3, strides=strides,
								  padding='same', use_bias=False)


class BasicBlock(tf.keras.Model):
	expansion = 1

	def __init__(self, inplanes, planes, strides=1, downsample=None):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(planes, strides)
		self.bn1 = tf.keras.layers.BatchNormalization()
		self.conv2 = conv3x3(planes)
		self.bn2 = tf.keras.layers.BatchNormalization()
		self.downsample = downsample
		self.strides = strides

	def call(self, x):
		residual = x

		out = tf.nn.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		return tf.nn.relu(out)


class Bottleneck(tf.keras.Model):
	expansion = 4

	def __init__(self, inplanes, planes, strides=1, downsample=None):
		super(Bottleneck, self).__init__()
		self.conv1 = tf.keras.layers.Conv2D(planes, kernel_size=1, use_bias=False)
		self.bn1 = tf.keras.layers.BatchNormalization()
		self.conv2 = tf.keras.layers.Conv2D(planes, kernel_size=3, strides=strides, padding="same", use_bias=False)
		self.bn2 = tf.keras.layers.BatchNormalization()
		self.conv3 = tf.keras.layers.Conv2D(planes * 4, kernel_size=1, use_bias=False)
		self.bn3 = tf.keras.layers.BatchNormalization()
		self.downsample = downsample
		self.strides = strides

	def call(self, x):
		residual = x

		out = tf.nn.relu(self.bn1(self.conv1(x)))
		out = tf.nn.relu(self.bn2(self.conv2(out)))
		out = self.bn3(self.conv3(out))

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		return tf.nn.relu(out)


###！ resnet18/ resnet34 didn't use pretrained model
class ResNet(tf.keras.Model):
	def __init__(self, block, layers, num_classes=3, scale=1):
		self.inplanes = 64
		super(ResNet, self).__init__()

		self.conv1 = tf.keras.layers.Conv2D(64, 7, 2, padding="same", input_shape=(640, 640, 3), data_format='channels_last', use_bias=False)
		self.bn1 = tf.keras.layers.BatchNormalization()
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], strides=2)
		self.layer3 = self._make_layer(block, 256, layers[2], strides=2)
		self.layer4 = self._make_layer(block, 512, layers[3], strides=2)
		# self.avgpool
		# self.fc

		# Top layer
		self.toplayer = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding="same")  # Reduce channels
		self.toplayer_bn = tf.keras.layers.BatchNormalization()

		# Smooth layers
		self.smooth1 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same")
		self.smooth1_bn = tf.keras.layers.BatchNormalization()

		self.smooth2 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same")
		self.smooth2_bn = tf.keras.layers.BatchNormalization()

		self.smooth3 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same")
		self.smooth3_bn = tf.keras.layers.BatchNormalization()

		# Lateral layers
		self.latlayer1 = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding="same")
		self.latlayer1_bn = tf.keras.layers.BatchNormalization()

		self.latlayer2 = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding="same")
		self.latlayer2_bn = tf.keras.layers.BatchNormalization()

		self.latlayer3 = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding="same")
		self.latlayer3_bn = tf.keras.layers.BatchNormalization()

		self.conv2 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same")
		self.bn2 = tf.keras.layers.BatchNormalization()
		self.conv3 = tf.keras.layers.Conv2D(num_classes, kernel_size=1, strides=1, padding="same")

		self.scale = scale

	def _make_layer(self, block, planes, blocks, strides=1):
		downsample = None
		if strides != 1 or self.inplanes != planes * block.expansion:
			downsample = tf.keras.Sequential([
				tf.keras.layers.Conv2D(planes * block.expansion,
									   kernel_size=1, strides=strides, use_bias=False),
				tf.keras.layers.BatchNormalization()
			])

		layers = []
		layers.append(block(self.inplanes, planes, strides, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return tf.keras.Sequential(layers)

	def _upsample(self, x, y, scale=1):
		_, H, W, _ = y.shape
		return tf.image.resize(x, (H // scale, W // scale))

	def _upsample_add(self, x, y):
		return self._upsample(x, y) + y

	def call(self, x):
		h = x
		h = tf.nn.relu(self.bn1(self.conv1(h)))
		h = tf.nn.max_pool2d(h, 3, 2, 'SAME', data_format='NHWC')

		h = self.layer1(h)
		c2 = h
		h = self.layer2(h)
		c3 = h
		h = self.layer3(h)
		c4 = h
		h = self.layer4(h)
		c5 = h

		# Top-down
		p5 = self.toplayer(c5)
		p5 = tf.nn.relu(self.toplayer_bn(p5))

		c4 = self.latlayer1(c4)
		c4 = tf.nn.relu(self.latlayer1_bn(c4))
		p4 = self._upsample_add(p5, c4)
		p4 = self.smooth1(p4)
		p4 = tf.nn.relu(self.smooth1_bn(p4))

		c3 = self.latlayer2(c3)
		c3 = tf.nn.relu(self.latlayer2_bn(c3))
		p3 = self._upsample_add(p4, c3)
		p3 = self.smooth2(p3)
		p3 = tf.nn.relu(self.smooth2_bn(p3))

		c2 = self.latlayer3(c2)
		c2 = tf.nn.relu(self.latlayer3_bn(c2))
		p2 = self._upsample_add(p3, c2)
		p2 = self.smooth3(p2)
		p2 = tf.nn.relu(self.smooth3_bn(p2))

		#       make p2,p3,p4,p5 have the same size
		p3 = self._upsample(p3, p2)
		p4 = self._upsample(p4, p2)
		p5 = self._upsample(p5, p2)
		# n=(n,h,w,c)
		out = tf.concat((p2, p3, p4, p5), 3)
		out = self.conv2(out)
		out = tf.nn.relu(self.bn2(out))
		out = self.conv3(out)
		out = self._upsample(out, x, scale=self.scale)
		# output 7 point/class
		return out


class TFPreResNet(tf.keras.Model):
	def __init__(self, layers, num_classes=7, scale=1):
		super(TFPreResNet, self).__init__()

		resnet = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet')
		count = 0
		act_ind = []  # acti layer index
		for subm in resnet.submodules:
			if isinstance(subm, tf.keras.layers.Activation):
				act_ind.append(count)
			count += 1

		a = 0
		outputs = []
		for i in layers:
			a += 3 * i
			outputs.append(resnet.layers[act_ind[a]].output)

		self.res = tf.keras.Model(inputs=resnet.input, outputs=outputs)
		# self.avgpool
		# self.fc

		# Top layer
		self.toplayer = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding="same")  # Reduce channels
		self.toplayer_bn = tf.keras.layers.BatchNormalization()

		# Smooth layers
		self.smooth1 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same")
		self.smooth1_bn = tf.keras.layers.BatchNormalization()

		self.smooth2 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same")
		self.smooth2_bn = tf.keras.layers.BatchNormalization()

		self.smooth3 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same")
		self.smooth3_bn = tf.keras.layers.BatchNormalization()

		# Lateral layers
		self.latlayer1 = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding="same")
		self.latlayer1_bn = tf.keras.layers.BatchNormalization()

		self.latlayer2 = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding="same")
		self.latlayer2_bn = tf.keras.layers.BatchNormalization()

		self.latlayer3 = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding="same")
		self.latlayer3_bn = tf.keras.layers.BatchNormalization()

		self.conv2 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same")
		self.bn2 = tf.keras.layers.BatchNormalization()
		self.conv3 = tf.keras.layers.Conv2D(num_classes, kernel_size=1, strides=1, padding="same")

		self.scale = scale

	def _upsample(self, x, y, scale=1):
		_, H, W, _ = y.shape
		return tf.image.resize(x, (H // scale, W // scale))

	def _upsample_add(self, x, y):
		return self._upsample(x, y) + y

	def call(self, x):
		h = x

		c2, c3, c4, c5 = self.res(h)

		# Top-down
		p5 = self.toplayer(c5)
		p5 = tf.nn.relu(self.toplayer_bn(p5))

		c4 = self.latlayer1(c4)
		c4 = tf.nn.relu(self.latlayer1_bn(c4))
		p4 = self._upsample_add(p5, c4)
		p4 = self.smooth1(p4)
		p4 = tf.nn.relu(self.smooth1_bn(p4))

		c3 = self.latlayer2(c3)
		c3 = tf.nn.relu(self.latlayer2_bn(c3))
		p3 = self._upsample_add(p4, c3)
		p3 = self.smooth2(p3)
		p3 = tf.nn.relu(self.smooth2_bn(p3))

		c2 = self.latlayer3(c2)
		c2 = tf.nn.relu(self.latlayer3_bn(c2))
		p2 = self._upsample_add(p3, c2)
		p2 = self.smooth3(p2)
		p2 = tf.nn.relu(self.smooth3_bn(p2))

		p3 = self._upsample(p3, p2)
		p4 = self._upsample(p4, p2)
		p5 = self._upsample(p5, p2)

		out = tf.concat((p2, p3, p4, p5), 3)
		out = self.conv2(out)
		out = tf.nn.relu(self.bn2(out))
		out = self.conv3(out)
		out = self._upsample(out, x, scale=self.scale)

		return out



def resnet18(pretrained=False, **kwargs):
	return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet50(pretrained=True, **kwargs):
	if pretrained:
		return TFPreResNet([3, 4, 6, 3], **kwargs)
	return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

#####################################################
### loss
def ohem_single(score, gt_text, training_mask):
	pos_num = (int)(np.sum(gt_text > 0.5)) - (int)(np.sum((gt_text > 0.5) & (training_mask <= 0.5)))

	if pos_num == 0:
		# selected_mask = gt_text.copy() * 0 # may be not good
		selected_mask = training_mask
		selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
		return selected_mask

	neg_num = (int)(np.sum(gt_text <= 0.5))
	neg_num = (int)(min(pos_num * 3, neg_num))

	if neg_num == 0:
		selected_mask = training_mask
		selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
		return selected_mask
	# print()
	neg_score = score[gt_text <= 0.5]
	neg_score_sorted = np.sort(-neg_score)
	threshold = -neg_score_sorted[neg_num - 1]

	selected_mask = ((score >= threshold) | (gt_text > 0.5)) & (training_mask > 0.5)
	selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
	return selected_mask

def ohem_batch(scores, gt_texts, training_masks):

	scores = scores.numpy()
	gt_texts = gt_texts.numpy()
	training_masks = training_masks.numpy()

	selected_masks = []
	for i in range(scores.shape[0]):
		selected_masks.append(ohem_single(scores[i, :, :], gt_texts[i, :, :], training_masks[i, :, :]))

	selected_masks = np.concatenate(selected_masks, 0)
	# selected_masks = torch.from_numpy(selected_masks).float()
	selected_masks = tf.convert_to_tensor(selected_masks, dtype=tf.float32)

	return selected_masks

def dice_loss(input, target, mask):
	# input = torch.sigmoid(input)
	input = tf.sigmoid(input)

	# input = input.contiguous().view(input.size()[0], -1)
	# target = target.contiguous().view(target.size()[0], -1)
	# mask = mask.contiguous().view(mask.size()[0], -1)
	input = tf.reshape(input, (input.shape[0], -1))
	target = tf.reshape(target, (target.shape[0], -1))
	mask = tf.reshape(mask, (mask.shape[0], -1))

	input = input * mask
	target = target * mask

	# a = torch.sum(input * target, 1)
	# b = torch.sum(input * input, 1) + 0.001
	# c = torch.sum(target * target, 1) + 0.001
	a = tf.reduce_sum(input * target, 1)
	b = tf.reduce_sum(input * input, 1) + 0.001
	c = tf.reduce_sum(target * target, 1) + 0.001

	d = (2 * a) / (b + c)
	# dice_loss = torch.mean(d)
	dice_loss = tf.reduce_mean(d)
	return 1 - dice_loss

def Dec_Loss_2(logites,gt_texts,gt_kernels,training_masks,kernal):
	'''
		 L = λLc + (1 − λ)Ls
		:param logites: shape: (batch, kernal_num, h, w) ,   tensor
		:param gt_texts: shape: (batch, h, w)  , np
		:param gt_kernels:shape: (batch, kernal_num - 1, h, w)   , np
		:param training_masks:shape: (batch, h, w)  , np
		:return: Loss
		'''
	with tf.name_scope("Loss"):
		pred_text = logites[:, 0, :, :]
		pred_kernals = logites[:, 1:, :, :]

		selected_masks = ohem_batch(pred_text, gt_texts, training_masks)

		loss_text = dice_loss(pred_text, gt_texts, selected_masks)

		loss_kernels = []
		mask0 = tf.sigmoid(pred_text).numpy()
		mask1 = training_masks.numpy()
		selected_masks = ((mask0 > 0.5) & (mask1 > 0.5)).astype('float32')
		# selected_masks = torch.from_numpy(selected_masks).float()
		selected_masks = tf.convert_to_tensor(selected_masks, dtype=tf.float32)
		# selected_masks = Variable(selected_masks.cuda())

		for i in range(kernal-1):
			kernel_i = pred_kernals[:, i, :, :]
			gt_kernel_i = gt_kernels[:, i, :, :]
			loss_kernel_i = dice_loss(kernel_i, gt_kernel_i, selected_masks)
			loss_kernels.append(loss_kernel_i)
		loss_kernel = sum(loss_kernels) / len(loss_kernels)

		loss = 0.7 * loss_text + 0.3 * loss_kernel

	return loss


if __name__ == '__main__':
	model = resnet50()
	model.build(input_shape=(None, 224, 224, 3))
	model.summary()
