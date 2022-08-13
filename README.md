# python-MCTA-Multiple-Camera-Tracking-Accuracy
  
Python version of multiple camera tracking accuracy (MCTA).
  
This code refers to the official evaluation toolkit on http://www.mct2014.com/Evaluation.html.
And rewrite it as python version.

The column of csv file is:
cid(camera id), fid(frame id), pid(person id), tl_x, tl_y, w, h

-----

### How to run?
	python evalMCTA.py --gt_path PATH/TO/GT --pred_path PATH/TO/PRED
  
### example
	python evalMCTA.py --gt_path example/gt.csv --pred_path example/pred.csv
  
-----
Thank you :)