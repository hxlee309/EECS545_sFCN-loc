import glob
import numpy as np
import pickle
import time
from PIL import Image

ground_truth = sorted(glob.glob("../train_label_500_jpg/train_label_*.jpg"), key = lambda name: int(name[35:-4]))
pred = sorted(glob.glob("../sfcn_loc_final_train/*.jpg"), key = lambda name: int(name[24:-4]))

count = 1
sum_accuracy = 0.0
pixel_num = 500.0 * 500.0
accuracy_total = []
recall_total = []
precision_total = []
history = {}

#Recall Calculation
lane_num_total = 0.0
lane_recall_total = 0.0

#Precision Calculation
pred_num_total = 0.0

for n in range(len(pred)):
	num_correct = 0.0
	gt_img = Image.open(ground_truth[n])
	pred_img = Image.open(pred[n])
	gt_img = np.array(gt_img)
	pred_img = np.array(pred_img)

	#Recall Calculation
	lane_num = 0.0
	lane_recall = 0.0

	#Precision Calculation
	pred_num = 0.0

	for i in range(gt_img.shape[0]):
		for j in range(gt_img.shape[1]):
			if gt_img[i][j] == pred_img[i][j][0]:
				num_correct += 1.0
			if gt_img[i][j] == 255:
				lane_num += 1.0
				lane_num_total += 1.0
				if pred_img[i][j][0] != 0:
					lane_recall += 1.0
					lane_recall_total += 1.0
			if pred_img[i][j][0] != 0:
				pred_num += 1.0
				pred_num_total += 1.0
	accuracy_temp = num_correct/pixel_num
	if lane_num != 0:
		recall_temp = lane_recall/lane_num
		print(lane_recall, lane_num)
	else:
		recall_temp = 1.0
	if pred_num != 0:
		precision_temp = lane_recall/pred_num
	else:
		precision_temp = 1.0
	accuracy_total.append(accuracy_temp)
	recall_total.append(recall_temp)
	precision_total.append(precision_temp)
	sum_accuracy += accuracy_temp
	print(count,pred[n],accuracy_temp,recall_temp,precision_temp)
	count += 1

n = len(pred)
accuracy = sum_accuracy/n
recall = lane_recall_total/lane_num_total
precision = lane_recall_total/pred_num_total

print("The testing recall is", recall)
print("The testing precision is", precision)

history["pred_total"] = pred_num_total
history["lane_recall"] = lane_recall_total
history["lane_total"] = lane_num_total
history["recall"] = recall_total
history["accuracy"] = accuracy_total
history["gt"] = ground_truth
history["pred"] = pred

with open('sfcn_testing_result.pickle', 'wb') as handle:
    pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Testing History is Saved!")