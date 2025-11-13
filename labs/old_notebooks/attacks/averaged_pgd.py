from typing import Union, Any, Callable

import eagerpy as ep
from foolbox import Model, Misclassification, TargetedMisclassification
from foolbox.attacks.base import T, raise_if_kwargs, get_criterion
from foolbox.attacks.projected_gradient_descent import LinfProjectedGradientDescentAttack
from secml.adv.attacks import CAttackEvasionFoolbox
from secml.adv.attacks.evasion.foolbox.losses.ce_loss import CELoss
from secml.adv.attacks.evasion.foolbox.secml_autograd import as_tensor
from secml.array import CArray


class AveragedPGD(LinfProjectedGradientDescentAttack):

    def __init__(self, *args, **kwargs):
        self.num_neighbors = kwargs.pop('k')
        self.sigma = kwargs.pop('sigma')
        super(AveragedPGD, self).__init__(*args, **kwargs)

    def get_loss_fn(
            self, model: Model, labels: ep.Tensor
    ) -> Callable[[ep.Tensor], ep.Tensor]:
        # can be overridden by users
        def loss_fn(x, labels):
            logits = model(x)

            c_minimize = labels
            c_maximize = best_other_classes(logits, labels)

            N = len(x)
            rows = range(N)

            logits_diffs = logits[rows, c_minimize] - logits[rows, c_maximize]
            assert logits_diffs.shape == (N,)

            return - logits_diffs

        return loss_fn

    def run(
            self,
            model: Model,
            inputs: T,
            criterion: Union[Misclassification, TargetedMisclassification, T],
            *,
            epsilon: float,
            **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        x0, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)
        del inputs, criterion, kwargs

        # perform a gradient ascent (targeted attack) or descent (untargeted attack)
        if isinstance(criterion_, Misclassification):
            gradient_step_sign = 1.0
            classes = criterion_.labels
        elif hasattr(criterion_, "target_classes"):
            gradient_step_sign = -1.0
            classes = criterion_.target_classes  # type: ignore
        else:
            raise ValueError("unsupported criterion")

        loss_fn = self.get_loss_fn(model, classes)

        if self.abs_stepsize is None:
            stepsize = self.rel_stepsize * epsilon
        else:
            stepsize = self.abs_stepsize

        if self.random_start:
            x = self.get_random_start(x0, epsilon)
            x = ep.clip(x, *model.bounds)
        else:
            x = x0

        loss = loss_fn(x, classes)  # required for having it in the path
        gradients = ep.zeros_like(x)
        for _ in range(self.steps):
            for k in range(self.num_neighbors):
                noise = ep.normal(x, shape=x.shape)

                pos_theta = x + self.sigma * noise
                neg_theta = x - self.sigma * noise

                pos_loss = loss_fn(pos_theta, classes)
                neg_loss = loss_fn(neg_theta, classes)

                gradients += (pos_loss - neg_loss) * noise

            # _, gradients = self.value_and_grad(loss_fn, x)
            gradients = self.normalize(gradients, x=x, bounds=model.bounds)
            x = x + gradient_step_sign * stepsize * gradients
            x = self.project(x, x0, epsilon)
            x = ep.clip(x, *model.bounds)

            loss = loss_fn(x, classes)  # required for having it in the path

        return restore_type(x)


def best_other_classes(logits: ep.Tensor, exclude: ep.Tensor) -> ep.Tensor:
    other_logits = logits - ep.onehot_like(logits, exclude, value=ep.inf)
    return other_logits.argmax(axis=-1)


class CFoolboxAveragedPGD(CELoss, CAttackEvasionFoolbox):

    def __init__(self, classifier, y_target=None, lb=0.0, ub=1.0,
                 epsilons=0.2,
                 rel_stepsize=0.025, abs_stepsize=None, steps=50,
                 k=100, sigma=8 / 255,
                 random_start=True):
        super(CFoolboxAveragedPGD, self).__init__(classifier, y_target,
                                                  lb=lb, ub=ub,
                                                  fb_attack_class=AveragedPGD,
                                                  epsilons=epsilons,
                                                  rel_stepsize=rel_stepsize,
                                                  abs_stepsize=abs_stepsize,
                                                  steps=steps,
                                                  random_start=random_start,
                                                  k=k, sigma=sigma)
        self._x0 = None
        self._y0 = None
        self.k = k
        self.sigma = sigma

    def _run(self, x, y, x_init=None):
        self._x0 = as_tensor(x)
        self._y0 = as_tensor(y)
        out, _ = super(CFoolboxAveragedPGD, self)._run(x, y, x_init)
        path_queries = CArray(range(0, self.x_seq.shape[0], (self.k * 2) + 1))
        self._x_seq = self.x_seq[path_queries, :]
        self._f_seq = self.objective_function(self.x_seq)
        f_opt = self.objective_function(out)
        return out, f_opt
