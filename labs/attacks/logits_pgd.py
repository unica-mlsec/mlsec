from typing import Callable

import eagerpy as ep
from foolbox import Model
from foolbox.attacks import LinfProjectedGradientDescentAttack
from secml.adv.attacks import CAttackEvasionFoolbox
from secml.adv.attacks.evasion.foolbox.losses.logits_loss import LogitsLoss
from secml.adv.attacks.evasion.foolbox.secml_autograd import as_tensor


class PGDLogitsLinf(LinfProjectedGradientDescentAttack):
    def get_loss_fn(
            self, model: Model, labels: ep.Tensor
    ) -> Callable[[ep.Tensor], ep.Tensor]:
        def loss_fn(inputs: ep.Tensor) -> ep.Tensor:
            rows = range(inputs.shape[0])
            logits = model(inputs)
            c_minimize = labels  # labels
            c_maximize = best_other_classes(logits, labels)
            loss = (logits[rows, c_maximize] - logits[rows, c_minimize]).sum()
            return loss

        return loss_fn


def best_other_classes(logits: ep.Tensor, exclude: ep.Tensor) -> ep.Tensor:
    other_logits = logits - ep.onehot_like(logits, exclude, value=ep.inf)
    return other_logits.argmax(axis=-1)


class CFoolboxLogitsPGD(LogitsLoss, CAttackEvasionFoolbox):
    __class_type = 'e-foolbox-logits-pgd'

    def __init__(self, classifier, y_target=None, lb=0.0, ub=1.0,
                 epsilons=0.2,
                 rel_stepsize=0.025, abs_stepsize=None, steps=50,
                 random_start=True):
        super(CFoolboxLogitsPGD, self).__init__(classifier, y_target,
                                                lb=lb, ub=ub,
                                                fb_attack_class=PGDLogitsLinf,
                                                epsilons=epsilons,
                                                rel_stepsize=rel_stepsize,
                                                abs_stepsize=abs_stepsize,
                                                steps=steps,
                                                random_start=random_start)
        self._x0 = None
        self._y0 = None
        self.confidence = 0

    def _run(self, x, y, x_init=None):
        self._x0 = as_tensor(x)
        self._y0 = as_tensor(y)
        out, _ = super(CFoolboxLogitsPGD, self)._run(x, y, x_init)
        f_opt = self.objective_function(out)
        return out, f_opt
