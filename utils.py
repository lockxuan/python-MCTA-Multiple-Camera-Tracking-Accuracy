import numpy as np






def IoUcost(groundtruth, trackresult):
	""" Calaulate the IoU cost between gt and pred.

	Parameters:
	-----------
	groundtruth: Dataframe
		Dataframe filtered by frame id & camera id
	trackresult: Dataframe
		Dataframe filtered by frame id & camera id

	Return:
	-------
	score: 2 x 2 array
		2x2 matrix of the IoU cost between gt and pred
	"""
	score = np.zeros((len(groundtruth), len(trackresult)))

	for i in range(len(groundtruth)):
		for j in range(len(trackresult)):
			score[i][j] = IoU(groundtruth.iloc[i], trackresult.iloc[j])

	return score



def IoU(gt, pred):
	""" Calaulate the IoU between gt(single bbox) and pred(single bbox).
		bbox is "tlwh" format now, can be changed by any format

	Parameters:
	-----------
	gt: Dataframe
		One row dataframe
	pred: Dataframe
		One row dataframe

	Return:
	-------
	IoU: float
		IoU of two bboxes.
	"""
	gt_tlwh = list(map(int, gt.values.tolist()[3:7]))
	pred_tlwh = list(map(int, pred.values.tolist()[3:7]))

	iou_x = max(gt_tlwh[0], pred_tlwh[0])
	iou_y = max(gt_tlwh[1], pred_tlwh[1])
	iou_w = max(0, min(gt_tlwh[2]+gt_tlwh[0], pred_tlwh[2]+pred_tlwh[0]) - iou_x)
	iou_h = max(0, min(gt_tlwh[3]+gt_tlwh[1], pred_tlwh[3]+pred_tlwh[1]) - iou_y)

	interarea = iou_w * iou_h
	unionarea = gt_tlwh[2] * gt_tlwh[3] + pred_tlwh[2] * pred_tlwh[3] - interarea

	if unionarea == 0:
		return 0

	IoU = interarea / unionarea
	return IoU








