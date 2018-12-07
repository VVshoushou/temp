#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.append('/home/vv/loglizer/')

import pandas as pd
import os
import numpy as np
import pickle
import utils.evaluation as ev

def anomaly_prediction(weigh_data, label_data, C, threshold):
	""" detect anomalies by projecting into a subspace with C

	Args:
	--------
	weigh_data: weighted raw data
	label_data: the labels list
	threshold: used as the threshold, which determines the anomalies
	C: the projection matrix

	Returns:
	--------

	"""
	print ('anomaly detecting with the history logs...')
	event_num, inst_size  = weigh_data.shape
	predict_results = np.zeros((inst_size),int)
	for i in range(inst_size):
		ya = np.dot(C,weigh_data[:,i])
		SPE = np.dot(ya,ya)
		if SPE > threshold:
			predict_results[i] = 1	#1 represent failure
	assert len(label_data) == len(predict_results)
	ev.evaluate(label_data, predict_results)



if __name__ == '__main__':
	"""
	!!!  test_dataset = [ weigh_data, labels]
	"""
	with open('saved_model/model.pickle','rb') as file:
		model = pickle.load(file)
		#print(model[0])
		C = model[0]
		threshold = model[1]

	with open('saved_model/testset.pickle','rb') as f1:
		testset = pickle.load(f1)
		weigh_data_all = testset[0]
		label_data_all = testset[1]

		#print(weigh_data_all.shape)
		weigh_data = weigh_data_all[:,:100]
		#print(weigh_data.shape)
		label_data = label_data_all[:100]
		#print(len(label_data))

	anomaly_prediction(weigh_data, label_data, C, threshold)
