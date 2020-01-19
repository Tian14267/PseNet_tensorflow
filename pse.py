# -*- coding:utf-8 -*-
import numpy as np
import cv2
import queue as Queue


def pse(kernals, min_area):
	kernal_num = len(kernals)
	pred = np.zeros(kernals[0].shape, dtype='int32')

	label_num, label = cv2.connectedComponents(kernals[kernal_num - 1], connectivity=4)  ## 最后一个 轮廓提取

	for label_idx in range(1, label_num):  #### 用于筛选面积大于min_area的区域
		if np.sum(label == label_idx) < min_area:
			label[label == label_idx] = 0

	queue = Queue.Queue(maxsize=0)
	next_queue = Queue.Queue(maxsize=0)
	points = np.array(np.where(label > 0)).transpose((1, 0))  ## 矩阵转置 (2,n) --> (n,2)
	# points_1 = np.array(np.where(label > 0))
	# points = points_1.transpose((1, 0))
	# print(points_1[0][0],points_1[1][0],"####",points[0])
	for point_idx in range(points.shape[0]):
		x, y = points[point_idx, 0], points[point_idx, 1]
		l = label[x, y]
		queue.put((x, y, l))  ### 把该点放队列中
		pred[x, y] = l

	new_kernals = []
	new_kernals.append(kernals[kernal_num - 1])
	new_kernals.append(kernals[0])
	dx = [-1, 1, 0, 0]
	dy = [0, 0, -1, 1]
	for kernal_idx in range(kernal_num - 2, -1, -1):
		# for kernal_idx in range(len(new_kernals)):
		# print("Kernal_idx: ",kernal_idx)
		kernal = kernals[kernal_idx].copy()
		while not queue.empty():
			(x, y, l) = queue.get()  ### 获取队列中的数据

			is_edge = True
			for j in range(4):
				tmpx = x + dx[j]
				tmpy = y + dy[j]
				if tmpx < 0 or tmpx >= kernal.shape[0] or tmpy < 0 or tmpy >= kernal.shape[1]:  ### 判断是否超出图像界面
					continue
				if kernal[tmpx, tmpy] == 0 or pred[tmpx, tmpy] > 0:  ###判断是否为边界
					continue

				queue.put((tmpx, tmpy, l))
				pred[tmpx, tmpy] = l
				is_edge = False
			if is_edge:
				next_queue.put((x, y, l))

		# kernal[pred > 0] = 0
		queue, next_queue = next_queue, queue

	# points = np.array(np.where(pred > 0)).transpose((1, 0))
	# for point_idx in range(points.shape[0]):
	#	 x, y = points[point_idx, 0], points[point_idx, 1]
	#	 l = pred[x, y]
	#	 queue.put((x, y, l))

	return pred
