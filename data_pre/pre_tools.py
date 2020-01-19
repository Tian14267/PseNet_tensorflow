# -*- coding:utf-8 -*-
import os
import numpy as np
import Polygon as plg
import pyclipper
import cv2

def read_image(image_path):
	try:
		image = cv2.imread(image_path) ## BGR
		#image = image[:, :, [2,1,0]] ##改为RGB
	except:
		print(image_path)
		raise
	return image

def get_box(gt_path,img):
	h, w = img.shape[0:2]
	f = open(gt_path,"r",encoding="utf-8")
	lines = f.readlines()
	bboxes = []
	tags = []  ### 用来标记box
	for line in lines:
		box = line.split(",")[:8]
		trag = line.split(",")[-1]

		if "#" in trag:
			tags.append(False)
		else:
			tags.append(True)

		b_box = [float(box[i]) for i in range(8)]
		#### 对 Box 进行归一化处理，方便后续调整图像尺寸时对Box再进行调整
		bbox = np.asarray(b_box) / ([w,h]*4)
		bboxes.append(bbox)
	return np.array(bboxes),tags

def scale(img, long_size=2240):
	h, w = img.shape[0:2]
	scale = long_size * 1.0 / max(h, w)
	img = cv2.resize(img, dsize=None, fx=scale, fy=scale)#比例因子，以scale比例放大
	return img

###########################################################
###   调整 Box 尺寸
def dist(a, b):
	return np.sqrt(np.sum((a - b) ** 2))

def perimeter(bbox):
	peri = 0.0
	for i in range(bbox.shape[0]):
		peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
	return peri

def box_shrink(boxes,rate,max_shr=20):
	rate = rate * rate
	shrinked_bboxes = []
	for box in boxes:
		area = plg.Polygon(box).area()  ## box 面积
		peri = perimeter(box)  ## box 周长

		pco = pyclipper.PyclipperOffset()
		pco.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
		offset = min((int)(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)

		shrinked_box = pco.Execute(-offset)
		if len(shrinked_box) == 0:
			shrinked_bboxes.append(box)
			continue

		shrinked_box = np.array(shrinked_box)[0]  ### list n
		if shrinked_box.shape[0] <= 2: ## shrinked_bbox.shape[0] = 4
			shrinked_bboxes.append(box)
			continue

		shrinked_bboxes.append(shrinked_box)

	return np.array(shrinked_bboxes)

#########################################################################

def batch_data(data_loader,batch_num,batch_size):
	'''
	用于生成一个batch的数据
	:param data_loader: Iterator to generate data
	:param batch_num: number of batch in iterator Now
	:param batch_size:
	:return: batch of data
	'''
	images = []
	gt_texts = []
	train_masks = []
	kernal_images = []
	for i in range(batch_size):
		data_count = (batch_num - 1) * batch_size + i
		img,gt_text,train_mask,kernal_img = data_loader[data_count]
		img = img.reshape((1,img.shape[0],img.shape[1],img.shape[2]))
		gt_text = gt_text.reshape((1,gt_text.shape[0],gt_text.shape[1]))
		train_mask = train_mask.reshape((1, train_mask.shape[0], train_mask.shape[1]))
		kernal_img = kernal_img.reshape((1, kernal_img.shape[0], kernal_img.shape[1], kernal_img.shape[2]))
		images.append(img)
		gt_texts.append(gt_text)
		train_masks.append(train_mask)
		kernal_images.append(kernal_img)

	images = np.concatenate(images,0) ## (8, 512, 512, 3)
	gt_texts = np.concatenate(gt_texts,0) ## (8, 512, 512)
	train_masks = np.concatenate(train_masks,0) ## (8, 512, 512)
	kernal_images = np.concatenate(kernal_images,0) ## (8, 6, 512, 512)

	return images,gt_texts,train_masks,kernal_images

def sigmod(x):
	return 1 / (1 + np.exp(-x))
##########################################################################
####  计算 ACC 和 IOU
class runningScore(object):

	def __init__(self, n_classes):
		self.n_classes = n_classes
		self.confusion_matrix = np.zeros((n_classes, n_classes))

	def _fast_hist(self, label_true, label_pred, n_class):
		### label_true,label_pred 的 max 值为 1
		mask = (label_true >= 0) & (label_true < n_class)

		#print("Max:",max(label_pred))
		if np.sum((label_pred[mask] < 0)) > 0:
			print (label_pred[label_pred < 0])
		score = n_class * label_true[mask].astype(np.int64) + label_pred[mask].astype(np.int64)
		hist = np.bincount(score, minlength=n_class**2).reshape(n_class, n_class)
		####  相当于画出一个柱状图。统计每个数字的出现次数
		####   true*2 + pred  : (0~2) + (0~1) = (0~3)
		return hist

	def update(self, label_trues, label_preds):
		# print label_trues.dtype, label_preds.dtype
		for lt, lp in zip(label_trues, label_preds): ### batch循环
			self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

	def get_scores(self):
		"""Returns accuracy score evaluation result.
			- overall accuracy
			- mean accuracy
			- mean IU
			- fwavacc
		"""
		hist = self.confusion_matrix
		acc = np.diag(hist).sum() / (hist.sum() + 0.0001)
		acc_cls = np.diag(hist) / (hist.sum(axis=1) + 0.0001)
		acc_cls = np.nanmean(acc_cls) ### 计算矩阵均值（跳过nan）
		iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 0.0001)
		mean_iu = np.nanmean(iu) ### 计算矩阵均值（跳过nan）
		freq = hist.sum(axis=1) / (hist.sum() + 0.0001)
		fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
		cls_iu = dict(zip(range(self.n_classes), iu))

		return {'Overall Acc': acc,
				'Mean Acc': acc_cls,
				'FreqW Acc': fwavacc,
				'Mean IoU': mean_iu,}, cls_iu

	def reset(self):
		self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

def calcult_acc(texts,gt_texts,training_masks,runningScore):
	'''
	计算 Acc
	:param texts: 预测出来的text：numpy格式
	:param gt_texts: 标注：numpy格式
	:param training_masks: 标注的mask：numpy格式
	:param runningScore: runningScore类。用条形图的形式计算acc
	:return: 返回 acc 字典 ：key包括('Overall Acc'，'Mean Acc'，'FreqW Acc'，'Mean IoU')
	'''
	def sigmod(x):
		return 1 / (1 + np.exp(-x))

	pred_text = sigmod(texts) * training_masks
	pred_text[pred_text <= 0.5] = 0
	pred_text[pred_text > 0.5] = 1

	gt_text = gt_texts * training_masks
	gt_text = gt_text.astype(np.int32)
	runningScore.update(gt_text, pred_text)
	score_text, _ = runningScore.get_scores()
	return score_text

