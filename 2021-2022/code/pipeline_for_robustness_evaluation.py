import argparse
import logging
import os

import pandas as pd
from robustbench.utils import load_model
from secml.adv.attacks import CFoolboxPGDLinf # noqa
from secml.array import CArray
from secml.data.loader import CDataLoaderCIFAR10
from secml.ml import CClassifierPyTorch

from utils import run_debug

# TODO take 2 models from the RobustBench library
"""
steps:
1. go to https://github.com/RobustBench/robustbench
2. search for the IDs that you want and add them to the list below
    - recommended: Standard and Engstrom2019Robustness
"""

MODEL_NAMES = ["Standard", "Engstrom2019Robustness"]

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int,
                    help='ID of the model to use. '
                         'Available models are: ' +
                         " ".join(
                             [f"{mname}({i})"
                              for i, mname in enumerate(MODEL_NAMES)]),
                    default=0, choices=list(range(len(MODEL_NAMES))))
parser.add_argument('--debug',
                    help='Runs the attack in debug mode. Used for '
                         'testing the loss function, and for '
                         'inspecting a single adversarial example.',
                    default=False, action='store_true')
parser.add_argument('--samples', type=int, help='Number of samples to use.', default=5)

args = parser.parse_args()

# create logger
logger = logging.getLogger('progress')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('progress.log')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

model_id = args.model
N_SAMPLES = args.samples
debug = args.debug
if debug:
    logging.warning("Keyword debug set to True, the number of samples will be ignored.")
    N_SAMPLES = 1
logger.info(f"Evaluating model {MODEL_NAMES[model_id]}...")


def prepare_data(n_samples=N_SAMPLES):
    """
    Prepares the dataset to use for the evaluation.
    :param n_samples: number of samples to include in the dataset.
    :return: the CDataset with the samples and labels.
    """
    _, ts = CDataLoaderCIFAR10().load()
    indexes = CArray.arange(ts.X.shape[0])
    indexes.shuffle()
    indexes = indexes[:n_samples]
    ts = ts[indexes, :]

    # this is to bring the samples in [0, 1]
    ts.X /= 255
    return ts


def prepare_model(model_name='Standard'):
    """
    Prepares the model to evaluate from the Robustbench Library.
    :param model_name: key for downloading the robustbench model. Can
        be found in https://github.com/RobustBench/robustbench.
    :return: the model wrapped inside a secml classifier.
    """
    model = load_model(model_name=model_name, dataset='cifar10',
                       threat_model='Linf')
    secml_model = CClassifierPyTorch(model, input_shape=(3, 32, 32), pretrained=True)
    return secml_model


def prepare_attack(clf, epsilon):
    """
    Creates the attack for the security evaluation pipeline
    :param clf: classifier to attack.
    :param epsilon: value for the constraint.
    :return: the instantiated attack
    """
    # TODO instantiate the PGD attack with Linf threat model
    """
    steps:
    1. go to https://github.com/RobustBench/robustbench and
        find out what epsilon constraint is used for the L-inf
        threat model
    2. instantiate the CFoolboxPGDLinf with 10 steps and the
        selected epsilon value (leave the other parameters as default)
    3. run the pipeline with the keyword --debug set and check if the
        parameters are working for the single example. If not, adjust
        the parameters
    """
    attack = CFoolboxPGDLinf(epsilons=epsilon, steps=10, classifier=clf)
    return attack


clf = prepare_model(MODEL_NAMES[model_id])
testing_points = prepare_data(n_samples=N_SAMPLES)
X, y = testing_points.X, testing_points.Y
attack = prepare_attack(clf, epsilon=8 / 255)

if debug:
    run_debug(clf, X, y, attack)
else:
    df = pd.DataFrame(columns=['label', 'pred', 'adv'])

    for sample in range(N_SAMPLES):
        x0, y0 = X[sample, :], y[sample]
        pred = clf.predict(x0)
        logger.info(f"Point {sample + 1}/{N_SAMPLES}")

        y_pred_adv, _, adv_ds, _ = attack.run(x0, y0)

        logger.debug(f"Orig. label: {y0.item()} Pred.: {pred.item()} "
                     f"Pred. after attack: {y_pred_adv.item()}")

        df = df.append({'label': y0.item(),
                        'pred': pred.item(),
                        'adv': y_pred_adv.item()}, ignore_index=True)
    logger.info("Evaluation complete. Storing results in csv report")

    if not os.path.exists("results"):
        os.mkdir("results")

    df.to_csv(f"results/{MODEL_NAMES[model_id]}.csv")
