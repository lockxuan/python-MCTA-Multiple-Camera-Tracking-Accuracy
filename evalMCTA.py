import pandas as pd
import numpy as np
import os, glob
import argparse
from tqdm import tqdm

from utils import IoUcost



def argument():
	parser = argparse.ArgumentParser()
	parser.add_argument('--gt_path', '--gt', default='PATH/TO/GT', type=str)
	parser.add_argument('--pred_path', '--pred', default='PATH/TO/PRED', type=str)
	parser.add_argument('--overlap_rate', default=0.5, type=float)

	args = parser.parse_args()
	return args





def load_data(gt_path, pred_path):
	""" loading groundtruths & predictions

	Parameters:
	-----------
	gt_path: str
		folder of groundtruth csv files
	pred_path: str
		folder of prediction csv files

	Return:
	-------
	gt_df: Dataframe
		groundtruth data, format: {cid, fid, pid, tl_x, tl_y, w, h}
	pred_df: Dataframe
		prediction data, format: {cid, fid, pid, tl_x, tl_y, w, h}
	# cid: camera id
	# fid: frame id
	# pid: person id by single camera tracking
	# bbox: tlwh
	"""

	gt = []
	pred = []

	allgt = glob.glob(os.path.join(gt_path, '*.csv'))
	allpred = glob.glob(os.path.join(pred_path, '*.csv'))
	for g in allgt:
		gt.append(pd.read_csv(g))
	for p in allpred:
		pred.append(pd.read_csv(p))

	assert len(gt) == len(pred)

	gt_df = pd.concat(gt)
	pred_df = pd.concat(pred)

	return gt_df, pred_df



def MCTA(trackData, groundTruth, overlap=0.5):
	evalPerformance = {}


	# only compute the trackData whos frame is in the groundtruth
	gtFrame = groundTruth['fid'].astype('category').unique()

	pre_map = [] # record last frame's [groundtruth trackinglabel] pairs

	# get the i frame's label data and tracking data
	test1 = 0
	test2 = 0

	# initialize parameters
	falsepos = 0
	missing = 0
	hypothesis = 0
	groundtruthes = 0
	mismatch_s = 0
	mismatch_c = 0
	truepos_s = 0
	truepos_c = 0
	precision = 1
	recall = 1

	for i, fid in enumerate(tqdm(gtFrame)):
		idxLabelData = groundTruth[groundTruth['fid'] == fid]
		idxTrackData = trackData[trackData['fid'] == fid]

		# tmp counter
		falsepos_tmp = 0		# false position
		missing_tmp = 0			# missing
		hypothesis_tmp = 0		# prediction
		groundtruthes_tmp = 0	# ground truth
		mismatch_s_tmp = 0		# miss match in 
		mismatch_c_tmp = 0		# miss match in 
		truepos_s_tmp = 0		# true position in
		truepos_c_tmp = 0		# true position in
		_map = []				#record [groundtruth trackinglabel] pairs


		# get the data for each camera
		for cid in groundTruth['cid'].astype('category').unique():
			idxLabelData_cam = idxLabelData[idxLabelData['cid'] == cid]
			idxTrackData_cam = idxTrackData[idxTrackData['cid'] == cid]

			count = 0
			# find the co-pair of groundtruth and the detection
			if not idxTrackData_cam.empty and not idxLabelData_cam.empty:
				#do the greedy alogrithm to find co-pair of groundtruth and detection

				# score is a [M N] distance matrix
				# co-pair is a flag matrix with size of [M N], M: number of groundtruth in frame i, N: number of tracking result in frame i
				# compute the distance between groundtruth and detection
				# compute the co-pair, just compute the co_pair based on score >= overlap
				score = IoUcost(idxLabelData_cam, idxTrackData_cam)
				co_pair = np.zeros((len(idxLabelData_cam), len(idxTrackData_cam)))
				for i, s in enumerate(score):
					midx = np.argmax(s)
					if s[midx] >= overlap:
						co_pair[i][midx] = 1
						count += 1
			else:
				score = []
				co_pair = []

			# compute the  number of truepos
			if count > 0:
				# compute the map of gt-tracking in current frame
				for i, pair in enumerate(co_pair):
					idx = np.where(pair == 1)[0]
					if len(idx) > 0:
						_map.append((idxLabelData_cam['pid'].iloc[i], idxTrackData_cam['pid'].iloc[idx[0]], cid))

		# compute the difference in map and pre-map
		hypothesis_tmp = len(idxTrackData)
		groundtruthes_tmp = len(idxLabelData)
		missing_tmp = groundtruthes_tmp - len(_map)
		falsepos_tmp = hypothesis_tmp - len(_map)

		# compute the difference in map and pre-map
		if _map:
			for m in _map:
				idswitchflag_s = 0	# handover idswsingle camera idsw
				idswitchflag_c = 0	# handover idsw
				trueposflag_s = 0

				if pre_map:
					_id = list(filter(lambda x:x[0] == m[0], pre_map))
					if len(_id) == 0:
						idd = list(filter(lambda x:x[1] == m[1], pre_map))
						if len(idd) > 0:
							idswitchflag_c = 1
							test1 += 1
					else:
						if _id[0][2] == m[2]:
							trueposflag_s = 1

						if _id[0][1] != m[1]:
							if _id[0][2] == m[2]:
								idswitchflag_s = 1
							else:
								idswitchflag_c = 1
								test2 += 1
						else:
							id1 = list(filter(lambda x:x[1] == m[1], _map))
							if _id[0][2] == m[2]:
								id2 = list(filter(lambda x:x[2] == m[2], id1))
								#if id2 and len(id2) != 1:
								if len(id2) > 1:
									idswitchflag_s = 1
							else:
								if id1 and len(id1) != 1:
								#if len(id1) > 1:
									idswitchflag_c = 1


				if idswitchflag_s == 1:
					mismatch_s_tmp += 1

				if idswitchflag_c == 1:
					mismatch_c_tmp += 1

				if trueposflag_s == 1:
					truepos_s_tmp += 1
				else:
					truepos_c_tmp += 1
		else:
			_map = []
			mismatch_s_tmp = 0
			mismatch_c_tmp = 0
			truepos_s_tmp = 0
			truepos_c_tmp = 0


		# Update parameters
		falsepos += falsepos_tmp
		missing += missing_tmp
		hypothesis += hypothesis_tmp
		groundtruthes += groundtruthes_tmp
		mismatch_s += mismatch_s_tmp
		mismatch_c += mismatch_c_tmp
		truepos_s += truepos_s_tmp
		truepos_c += truepos_c_tmp

		# map to pre-map
		if  len(_map) == 0:
			pre_map = pre_map
		elif len(pre_map) == 0:
			pre_map = _map

		if len(_map) > 0 and len(pre_map) > 0:
			# first step, find the intersection of map and pre_map
			# second step, update the intersection
			new_pre_map = [case for case in _map]
			gt_map = [gt for (gt, _, _) in _map]
			for (gt1, pred1, cam1) in pre_map:
				if gt1 not in gt_map:
					new_pre_map.append((gt1, pred1, cam1))

			pre_map = new_pre_map


	# output the parameter
	precision = 1 - falsepos/hypothesis
	recall = 1 - missing/groundtruthes
	print('precision', precision)
	print('recall', recall)
	print('gt', groundtruthes)
	print('hypothesis',hypothesis)
	print('missing', missing)
	print('falsepos',falsepos)
	print('mismatch_s',mismatch_s)
	print('mismatch_c',mismatch_c)
	print('truepos_s', truepos_s)
	print('truepos_c',truepos_c)
	print('')

	evalPerformance['gt'] = groundtruthes
	evalPerformance['results'] = hypothesis
	evalPerformance['missing'] = missing
	evalPerformance['falsepos'] = falsepos
	evalPerformance['mismatch_s'] = mismatch_s
	evalPerformance['mismatch_c'] = mismatch_c
	evalPerformance['truepos_s'] = truepos_s
	evalPerformance['truepos_c'] = truepos_c
	evalPerformance['precision'] = precision
	evalPerformance['recall'] = recall
	evalPerformance['mcta'] = (2*precision*recall/(precision+recall))*(1-mismatch_c/truepos_c)*(1-mismatch_s/truepos_s)

	return evalPerformance


				




def main(args):

	gt, pred = load_data(args.gt_path, args.pred_path)

	scores = MCTA(pred, gt, args.overlap_rate)
	print(scores)





if __name__ == '__main__':
	args = argument()
	main(args)




