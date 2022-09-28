from secml.core.constants import inf


def run_debug(clf, X, y, attack):
	"""
	Visualizes the image of the input sample and the perturbed sample,
	along with the debugging information for the optimization.
	:param clf: instantiated secml classifier
	:param X: initial sample
	:param y: label of the sample X
	:param attack: instantiated attack from the secml library
	:return: None
	"""
	dataset_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
	x0, y0 = X[0, :], y[0]
	y_pred_adv, _, adv_ds, _ = attack.run(x0, y0)
	from secml.figure import CFigure
	img_normal = x0.tondarray().reshape((3, 32, 32)).transpose(1, 2, 0)
	img_adv = adv_ds.X[0, :].tondarray().reshape((3, 32, 32)).transpose(1, 2, 0)

	diff_img = img_normal - img_adv
	diff_img -= diff_img.min()
	diff_img /= diff_img.max()

	fig = CFigure(height=7, width=15)
	fig.subplot(1, 3, 1)
	fig.sp.imshow(img_normal)
	fig.sp.title('{0}'.format(dataset_labels[y0.item()]))
	fig.sp.xticks([])
	fig.sp.yticks([])

	fig.subplot(1, 3, 2)
	fig.sp.imshow(img_adv)
	fig.sp.title('{0}'.format(dataset_labels[y_pred_adv.item()]))
	fig.sp.xticks([])
	fig.sp.yticks([])

	fig.subplot(1, 3, 3)
	fig.sp.imshow(diff_img)
	fig.sp.title('Amplified perturbation')
	fig.sp.xticks([])
	fig.sp.yticks([])
	fig.tight_layout()

	# visualize the attack loss
	attack_loss = attack.objective_function(attack.x_seq)
	fig_loss = CFigure()
	fig_loss.sp.plot(attack_loss)
	fig_loss.sp.title("Attack loss")

	# visualize the perturbation size
	pert_size = (attack.x_seq - x0).norm_2d(axis=1, order=inf)
	fig_pert_size = CFigure()
	fig_pert_size.sp.plot(pert_size)
	fig_pert_size.sp.title("Perturbation size (L-inf)")

	# visualize the logits of all the classes
	fig_scores = CFigure()
	for cl_idx, cl in enumerate(dataset_labels):
		scores = clf.decision_function(attack.x_seq, cl_idx)
		fig_scores.sp.plot(scores, label=cl)
	fig_scores.sp.title("Scores")
	fig_scores.sp.legend()

	CFigure.show()
