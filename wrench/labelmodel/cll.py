# https://proceedings.mlr.press/v161/arachie21a/arachie21a.pdf
# code taken from https://github.com/VTCSML/Constrained-Labeling-for-Weakly-Supervised-Learning/tree/main
import logging
from typing import Any, Optional, Union, Callable

from ..evaluation import METRIC
import numpy as np

# from ..basemodel import BaseLabelModel
# from ..dataset import BaseDataset

logger = logging.getLogger(__name__)

ABSTAIN = -1


class CLL:
    def __init__(self, **kwargs: Any):
        # super().__init__()
        self.n_class = None

    def fit(
        self,
        dataset_train: np.ndarray,  # Union[BaseDataset, np.ndarray],
        rule_error_init: float = 0.01,
        n_class: Optional[int] = None,
        weak: Optional[int] = None,
        n_weaks: Optional[int] = None,
        **kwargs: Any,
    ):
        L_train = dataset_train[0]
        self.n_class = n_class or int(np.max(L_train)) + 1
        self.p = L_train.shape[1]
        self.n = L_train.shape[0]
        self.L_train = L_train

        weak_errors = np.ones((self.p, self.n_class)) * rule_error_init
        # weak_signals should have shape p, n, k
        # L_train has shape n, p
        weak_signals = L_train.transpose()
        abstain_mask = weak_signals == -1
        pred_mask = ~abstain_mask
        expanded_abstain = np.tile(
            np.expand_dims(abstain_mask * -1.0, axis=2), (1, 1, self.n_class)
        )
        # we will get 0's for both abstain and first label, but we can use pred_mask to differentiate
        masked_preds = weak_signals * pred_mask
        # list elements should be k x n
        preds_expanded_list = [
            np.eye(self.n_class)[masked_preds[i, :]] for i in range(self.p)
        ]
        preds_expanded_list = [
            np.expand_dims(ele, axis=0) for ele in preds_expanded_list
        ]
        # now stack these all together
        weak_signals = np.vstack(preds_expanded_list)
        # mask away the abstentions
        pred_mask_expanded = np.tile(
            np.expand_dims(~abstain_mask, axis=2), (1, 1, self.n_class)
        )
        weak_signals = weak_signals * pred_mask_expanded + expanded_abstain

        constraints = self.set_up_constraint(weak_signals, weak_errors)
        constraints["weak_signals"] = weak_signals
        self.cll_preds = self.train_algorithm(constraints)

    # maintain function signature so runner will just work
    def predict_proba(
        self, dataset: np.ndarray
    ) -> np.ndarray:  # Union[BaseDataset, np.ndarray]) -> np.ndarray:
        return self.cll_preds

    def test(
        self,
        dataset: np.ndarray,
        metric_fn: Union[Callable, str],
        y_true: Optional[np.ndarray] = None,
        **kwargs,
    ):
        if isinstance(metric_fn, str):
            metric_fn = METRIC[metric_fn]
        if y_true is None:
            try:
                y_true = np.squeeze(dataset[1])
            except:
                y_true = np.array(dataset.labels)
        probas = self.predict_proba(dataset, **kwargs)
        return metric_fn(y_true, probas)

    def train_algorithm(self, constraint_set):
        """
        Trains CLL algorithm

        :param constraint_set: dictionary containing error constraints of the weak signals
        :return: average of learned labels over several trials
        :rtype: ndarray
        """
        constraint_set["constraints"] = ["error"]
        weak_signals = constraint_set["weak_signals"]
        assert len(weak_signals.shape) == 3, (
            "Reshape weak signals to num_weak x num_data x num_class"
        )
        _, n, k = weak_signals.shape
        # initialize y
        y = np.random.rand(n, k)
        # initialize hyperparameters
        rho = 0.1

        return self.run_constraints(y, rho, constraint_set)

    def run_constraints(self, y, rho, constraint_set, iters=300, enable_print=False):
        # Run constraints from CLL

        constraint_keys = constraint_set["constraints"]
        n, k = y.shape
        rho = n
        grad_sum = 0

        for iteration in range(iters):
            print_constraints = [iteration]
            print_builder = "Iteration %d, "
            constraint_viol = []
            viol_text = ""

            for key in constraint_keys:
                current_constraint = constraint_set[key]
                a_matrix = current_constraint["A"]
                bounds = current_constraint["b"]

                # get bound loss for constraint
                loss = self.bound_loss(y, a_matrix, bounds)
                # update constraint values
                constraint_set[key]["bound_loss"] = loss

                violation = np.linalg.norm(loss.clip(min=0))
                print_builder += key + "_viol: %.4e "
                print_constraints.append(violation)

                viol_text += key + "_viol: %.4e "
                constraint_viol.append(violation)

            y_grad = self.y_gradient(y, constraint_set)
            grad_sum += y_grad**2
            y = y - y_grad / np.sqrt(grad_sum + 1e-8)
            y = np.clip(y, a_min=0, a_max=1)

            constraint_set["violation"] = [viol_text, constraint_viol]
            if enable_print:
                print(print_builder % tuple(print_constraints))
        return y

    def bound_loss(self, y, a_matrix, bounds):
        """
        Computes the gradient of lagrangian inequality penalty parameters

        :param y: size (num_data, num_class) of estimated labels for the data
        :type y: ndarray
        :param a_matrix: size (num_weak, num_data, num_class) of a constraint matrix
        :type a_matrix: ndarray
        :param bounds: size (num_weak, num_class) of the bounds for the constraint
        :type bounds: ndarray
        :return: loss of the constraint (num_weak, num_class)
        :rtype: ndarray
        """
        constraint = np.zeros(bounds.shape)
        n, k = y.shape

        for i, current_a in enumerate(a_matrix):
            constraint[i] = np.sum(current_a * y, axis=0)
        return constraint - bounds

    def y_gradient(self, y, constraint_set):
        """
        Computes y gradient
        """
        constraint_keys = constraint_set["constraints"]
        gradient = 0

        for key in constraint_keys:
            current_constraint = constraint_set[key]
            a_matrix = current_constraint["A"]
            bound_loss = current_constraint["bound_loss"]

            for i, _ in enumerate(a_matrix):
                constraint = a_matrix[i]
                gradient += 2 * constraint * bound_loss[i]
        return gradient

    def set_up_constraint(self, weak_signals, error_bounds):
        """Set up error constraints for A and b matrices"""

        constraint_set = dict()
        m, n, k = weak_signals.shape
        precision_amatrix = np.zeros((m, n, k))
        error_amatrix = np.zeros((m, n, k))
        constants = []

        for i, weak_signal in enumerate(weak_signals):
            active_signal = weak_signal >= 0
            precision_amatrix[i] = (
                -1
                * weak_signal
                * active_signal
                / (np.sum(active_signal * weak_signal, axis=0) + 1e-8)
            )
            error_amatrix[i] = (1 - 2 * weak_signal) * active_signal

            # error denom to check abstain signals
            error_denom = np.sum(active_signal, axis=0) + 1e-8
            error_amatrix[i] /= error_denom

            # constants for error constraints
            constant = (weak_signal * active_signal) / error_denom
            constants.append(constant)

        # set up error upper bounds constraints
        constants = np.sum(constants, axis=1)
        assert len(constants.shape) == len(error_bounds.shape)
        bounds = error_bounds - constants
        error_set = self.build_constraints(error_amatrix, bounds)
        constraint_set["error"] = error_set

        return constraint_set

    def build_constraints(self, a_matrix, bounds):
        """params:
        a_matrix left hand matrix of the inequality size: num_weak x num_data x num_class type: ndarray
        bounds right hand vectors of the inequality size: num_weak x num_data type: ndarray
        return:
        dictionary containing constraint vectors
        """

        m, n, k = a_matrix.shape
        assert (m, k) == bounds.shape, "The constraint matrix shapes don't match"

        constraints = dict()
        constraints["A"] = a_matrix
        constraints["b"] = bounds
        return constraints
