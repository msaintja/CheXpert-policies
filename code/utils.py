import os
import time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score

class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def compute_batch_avg_auc(output, target):
	"""Computes the accuracy for a batch"""
	with torch.no_grad():

		output_np = output.cpu().numpy()
		target_np = target.cpu().numpy()

		output_np_lb = output_np.copy()
		target_np_lb = target_np.copy()
		# output_np_ub = output_np.copy()
		# target_np_ub = target_np.copy()

		output_np_lb[(target_np != 0) & (target_np != 1)] = 1
		target_np_lb[(target_np != 0) & (target_np != 1)] = 0

		# output_np_ub[(target_np != 0) & (target_np != 1)] = 1
		# target_np_ub[(target_np != 0) & (target_np != 1)] = 1

		pc_output_np_lb = output_np_lb.copy().T
		pc_target_np_lb = target_np_lb.copy().T
		# pc_output_np_ub = output_np_ub.copy().T
		# pc_target_np_ub = target_np_ub.copy().T


		# per sample AUC
		# output_np_lb = np.hstack((target_np_lb, [[1, 0]]*len(target_np)))
		# target_np_lb = np.hstack((target_np_lb, [[0, 1]]*len(target_np)))

		# output_np_ub = np.hstack((target_np_ub, [[1, 0]]*len(target_np)))
		# target_np_ub = np.hstack((target_np_ub, [[1, 0]]*len(target_np)))

		# correction_lb = target_np_lb.shape[1]*1.0/(target_np_lb.shape[1] - 2)

		# lb = np.mean([correction_lb * roc_auc_score(t, o, 'weighted') for t,o in zip(target_np_lb, output_np_lb)])

		# ub = np.mean([roc_auc_score(t, o, 'weighted') for t,o in zip(target_np_ub, output_np_ub)])

		# per class AUC
		pc_output_np_lb = np.hstack((pc_output_np_lb, [[1, 0]]*target_np.shape[1]))
		pc_target_np_lb = np.hstack((pc_target_np_lb, [[0, 1]]*target_np.shape[1]))

		# pc_output_np_ub = np.hstack((pc_target_np_ub, [[1, 0]]*target_np.shape[1]))
		# pc_target_np_ub = np.hstack((pc_target_np_ub, [[1, 0]]*target_np.shape[1]))

		pc_correction_lb = pc_target_np_lb.shape[1]*1.0/(pc_target_np_lb.shape[1] - 2)

		pc_lb = [pc_correction_lb * roc_auc_score(t, o, 'weighted') for t,o in zip(pc_target_np_lb, pc_output_np_lb)]

		# pc_ub = [roc_auc_score(t, o, 'weighted') for t,o in zip(pc_target_np_ub, pc_output_np_ub)]

		lb = np.mean(pc_lb)

		return lb, pc_lb #lb, ub, pc_lb, pc_ub


def train(model, device, data_loader, criterion, optimizer, epoch, print_freq=10):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	auc_lb = AverageMeter()
	# auc_ub = AverageMeter()

	model.train()

	end = time.time()
	for i, (input, target) in enumerate(data_loader):
		# measure data loading time
		data_time.update(time.time() - end)

		if isinstance(input, tuple):
			input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
		else:
			input = input.to(device)
		target = target.to(device)

		optimizer.zero_grad()
		output = model(input)
		loss = criterion(output, target)
		assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

		loss.backward()
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		losses.update(loss.item(), target.size(0))
		# auc_l, auc_u, pc_auc_l, pc_auc_u = compute_batch_avg_auc(output, target)
		auc_l, pc_auc_l = compute_batch_avg_auc(output, target)
		auc_lb.update(auc_l, target.size(0))
		# auc_ub.update(auc_u, target.size(0))

		if i % print_freq == 0:
			print('Epoch: [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Avg. Batch AUC {auclb.val:.3f} ({auclb.avg:.3f})\t'
				#   'Avg. Batch AUC (UB) {aucub.val:.3f} ({aucub.avg:.3f})\t'
				  'Per-class AUC {pcauclb}\t'
				#   'Per-class AUC (UB) {pcaucub}\t'
				  .format(
					epoch, i, len(data_loader), batch_time=batch_time,
					data_time=data_time, loss=losses, auclb=auc_lb, 
					# aucub=auc_ub, 
					pcauclb = np.around(pc_auc_l, 3) 
					# pcaucub = np.around(pc_auc_u, 3)
				))

	return losses.avg, auc_lb.avg


def evaluate(model, device, data_loader, criterion, print_freq=10):
	batch_time = AverageMeter()
	losses = AverageMeter()
	auc_lb = AverageMeter()
	# auc_ub = AverageMeter()

	results = []
	avg_aucs = []

	model.eval()

	with torch.no_grad():
		end = time.time()
		for i, (input, target) in enumerate(data_loader):

			if isinstance(input, tuple):
				input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
			else:
				input = input.to(device)
			target = target.to(device)

			output = model(input)
			loss = criterion(output, target)

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			losses.update(loss.item(), target.size(0))
			# auc_l, auc_u, pc_auc_l, pc_auc_u = compute_batch_avg_auc(output, target)
			auc_l, pc_auc_l = compute_batch_avg_auc(output, target)
			auc_lb.update(auc_l, target.size(0))
			avg_aucs.append(pc_auc_l)
			# auc_ub.update(auc_u, target.size(0))

			y_true = target.detach().to('cpu').numpy().tolist()
			y_pred = output.round().detach().to('cpu').numpy().tolist()
			results.extend(list(zip(y_true, y_pred)))

			if i % print_freq == 0:
				print('Test: [{0}/{1}]\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					  'Avg. Batch AUC {auclb.val:.3f} ({auclb.avg:.3f})\t'
					#   'Avg. Batch AUC (UB) {aucub.val:.3f} ({aucub.avg:.3f})\t'
					  'Per-class AUC {pcauclb} ({pcauclb_avg})\t'
					#   'Per-class AUC (UB) {pcaucub}\t'
					  .format(
					i, len(data_loader), batch_time=batch_time, loss=losses, auclb=auc_lb, 
					# aucub=auc_ub, 
					pcauclb = np.around(pc_auc_l, 3), 
					# pcaucub = np.around(pc_auc_u, 3)
					pcauclb_avg = np.around(np.mean(avg_aucs, axis = 0), 3)
					))

	return losses.avg, auc_lb.avg, results


def make_kaggle_submission(list_id, list_prob, path):
	if len(list_id) != len(list_prob):
		raise AttributeError("ID list and Probability list have different lengths")

	os.makedirs(path, exist_ok=True)
	output_file = open(os.path.join(path, 'my_predictions.csv'), 'w')
	output_file.write("SUBJECT_ID,MORTALITY\n")
	for pid, prob in zip(list_id, list_prob):
		output_file.write("{},{}\n".format(pid, prob))
	output_file.close()
