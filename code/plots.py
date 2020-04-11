import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.preprocessing import normalize
# TODO: You can use other packages if you want, e.g., Numpy, Scikit-learn, etc.


def plot_learning_curves(train_losses, valid_losses, train_auc_lbs, valid_auc_lbs):
	# TODO: Make plots for loss curves and accuracy curves.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.

	plt.figure()
	plt.plot(np.arange(len(train_losses)), train_losses, label='Train', color='blue')
	plt.plot(np.arange(len(valid_losses)), valid_losses, label='Validation', color='red')
	plt.ylabel('Loss')
	plt.xlabel('epoch')
	plt.legend(loc="best")
	plt.savefig("CheXpert-model_learning_curve_loss.png")

	train_auc = np.array(train_auc_lbs)
	valid_auc = np.array(valid_auc_lbs)


	plt.figure()
	plt.plot(np.arange(len(train_auc)), train_auc, label='Train')
	# plt.gca().fill_between(np.arange(len(train_auc)), train_auc_lbs, train_auc_ubs, facecolor='blue', alpha=0.5)
	plt.plot(np.arange(len(valid_auc)), valid_auc, label='Validation')
	# plt.gca().fill_between(np.arange(len(valid_auc)), valid_auc_lbs, valid_auc_ubs, facecolor='red', alpha=0.5)
	plt.ylabel('Avg. AUC')
	plt.xlabel('epoch')
	plt.legend(loc="best")
	plt.savefig("CheXpert-model_learning_curve_auc.png")



def plot_confusion_matrix(results, class_names):
	# TODO: Make a confusion matrix plot.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.

	results = np.array(results)
	matrices = multilabel_confusion_matrix(results[:,0], results[:,1])
	matrices = multilabel_confusion_matrix(results[:,0], results[:,1])

	for index, name in enumerate(class_names): 
		matrix = normalize(matrices[index], norm="l1")

		fig, ax = plt.subplots()
		heatmap = ax.pcolor(matrix, cmap=plt.cm.Blues, vmin=0, vmax=1)

		ax.set_xticks(np.arange(len(["NO", "YES"])) + 0.5)
		ax.set_yticks(np.arange(len(["NO", "YES"])) + 0.5)
		ax.set_xticklabels(["NO", "YES"])
		ax.set_yticklabels(["NO", "YES"])

		plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

		for i in range(len(["NO", "YES"])):
			for j in range(len(["NO", "YES"])):
				text = ax.text(j + 0.5, i + 0.5, np.round(matrix[i, j], 2), ha="center", va="center")

		ax.invert_yaxis()

		plt.colorbar(heatmap)

		ax.set_title("Normalized Confusion Matrix ({})".format(name))
		plt.ylabel('True')
		plt.xlabel('Predicted')
		fig.tight_layout()
		plt.savefig("CheXpert-model_confusion_matrix_{}.png".format(name))
