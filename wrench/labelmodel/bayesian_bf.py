# pylint: disable C0114
import logging
from typing import Any, Optional, Union

import cvxpy as cp
import numpy as np
import scipy as sp

from ..basemodel import BaseLabelModel
from ..dataset import BaseDataset

from .majority_voting import MajorityVoting
# from ..dataset.utils import check_weak_labels

generic_logger = logging.getLogger(__name__)

ABSTAIN = -1
# pylint: disable=C0103
# pylint: disable=C0115
# pylint: disable=W0612
# pylint: disable=R0914
# pylint: disable=R0902
class Bayesian_BF(BaseLabelModel):
    def __init__(self,
                 solver: Optional[str] = 'MOSEK',
                 verbose: Optional[bool] = False,
                 **kwargs: Any):
        super().__init__()
        self.solver = solver
        self.verbose = verbose

        ### create all the attributes and set to None to be 'pythonic'

        # logger
        self.logger = None
        # whether or not to add the majority vote classifier as a constraint
        self.constraint_type = None
        # whether or not to use equality constraints.  if equality constraints
        # are used, then the raw estimates of class frequencies, accuracies/
        # class accuracies/confusion matrices are used.  specifically, the
        # values in self.class_freq_probs[1, :] and self.param_probs[1, :]
        self.n_class = None
        # number of rules
        self.p = None
        # number of training (unlabeled) datapoints
        self.n_pts = None
        # number of predictions each rule makes (on the training dataset)
        self.n_preds_per_rule = None
        # number of datapoints to be used to estimate classifier parameters
        self.n_max_labeled = None
        # number of rules that have at least 1 labeled point
        self.n_rules_used = None
        # how many labeled points each rule got
        self.avg_labeled_per_rule = None
        # if unsupervised, whether to use 'majority_vote' or 'uniform'
        # initialization for class distribution prior
        self.unsup_balance_init = None
        # if using majority vote, whether to scale the resulting prior
        # hyperparameters so that it's like only 10 labeled points were used
        # i.e. you can get alpha_0=800, alpha_1=200 for your prior
        # hyperparams via majority vote -- the default below would scale it
        # to alpha_0=8, alpha_1=2
        self.unsup_balance_init_rescale = None
        # whether the rule accuracy/class distribution hyperparameters should be
        # initialized as following: start from beta(1,1), dirichlet(1,1,...)
        # and add 1 to each argument depending on whether the prediction was
        # correct or what the point was labeled
        self.labeled_params_unif_prior = None
        # whether to scale the hyperparameters for class distribution/rule
        # accuracy to the inputted number.  I.e. setting this to 10 will make
        # the beta/dirichlet hyperparameters add to 10.
        self.labeled_params_scale = None
        # array of dirichlet distribution parameters matching the empirical
        # mean and variance of the class distribution
        self.class_freq_params = None
        # array of arrays.  each item is a [length 2 array of beta distribution
        # parameters (accuracy constraints) or kxk matrix where each row is
        # a dirichlet's hyperparameters (class_accuracy or confusion matrix
        # constraints)]
        self.rule_params = None
        # CVXPY parameters for the convex program that checks if a non-empty
        # polytope is formed
        self.class_freq_param = None
        self.acc_param = None
        # the convex program for finding the posterior mode
        self.mode_prob = None
        # the status of the above problem
        self.mode_prob_status = None
        # the actual posterior mode labeling
        self.post_mode_labeling = None
        # the convex program for finding the weights for the posterior mode pred
        self.wt_prob = None
        # the status of the above problem
        self.wt_prob_status = None
        # weights used for prediction, gotten from solving wt_prob
        self.param_wts = None
        self.class_freq_wts = None

    def fit(self,
            dataset_train: Union[BaseDataset, np.ndarray],
            dataset_valid: Union[BaseDataset, np.ndarray],
            y_valid: Optional[np.ndarray] = None,
            n_class: Optional[int] = None,
            constraint_type = 'accuracy',
            n_max_labeled: Optional[int] = -1,
            unsup_balance_init: Optional[str] = 'majority_vote',
            unsup_balance_init_rescale: Optional[int] = 10,
            labeled_params_unif_prior: Optional[bool] = True,
            labeled_params_scale: Optional[int] = -1,
            # force_train_as_labeled_dataset: Optional[bool] = False,
            logger = generic_logger,
            **kwargs: Any):
            # balance: Optional[np.ndarray] = None,
            # weak: Optional[int] = None,
            # n_weaks: Optional[int] = None,
            # seed: Optional[int] = None,
            # random_guess: Optional[int] = None,

        self.unsup_balance_init = unsup_balance_init
        self.unsup_balance_init_rescale = unsup_balance_init_rescale
        self.labeled_params_unif_prior = labeled_params_unif_prior,
        self.labeled_params_scale = labeled_params_scale
        self.constraint_type = constraint_type
        self.n_max_labeled = n_max_labeled
        self.logger=logger

        # if constraint_type not in ['accuracy', 'class_accuracy',\
        #         'confusion_matrix']:
        if constraint_type not in ['accuracy']:
            raise ValueError(f"constraint_type argument ({constraint_type})"
                    f" must be 'accuracy'.")

        if n_max_labeled < 1 and\
                unsup_balance_init not in ['uniform', 'majority_vote']:
            raise ValueError(f"unsup_balance_init argument"
                    f" ({unsup_balance_init}) must be in "
                    f"['uniform', 'majority_vote']")

        if unsup_balance_init_rescale == 0:
            raise ValueError("unsup_balance_init_rescale must be -1 or > 0")

        # self._update_hyperparas(**kwargs)
        if isinstance(dataset_train, BaseDataset):
            if n_class is not None:
                assert n_class == dataset_train.n_class
            else:
                n_class = dataset_train.n_class

        # get ground truths for each of the datasets
        if y_valid is None:
            y_valid = np.squeeze(dataset_valid[1])
            # y_valid = np.array(dataset_valid.labels)

        # y_train = np.squeeze(dataset_train[1])

        L = dataset_train[0]
        L_val = dataset_valid[0]
        # L = check_weak_labels(dataset_train, n_weaks=n_weaks,\
        #            random_guess=random_guess)
        # L_val = check_weak_labels(dataset_valid, n_weaks=n_weaks,\
        #            random_guess=random_guess)

        # number of classes
        n_class = n_class or round(L.max()) + 1
        self.n_class = n_class

        # get one hot encodings of the predictions
        L_aug = self._initialize_L_aug(L)

        # classifier count, number of datapoints
        self.p, self.n_pts, _ = L_aug.shape

        # count number of predictions for every classifier
        n_preds_per_rule = np.sum(L_aug, axis=(-2, -1))
        self.n_preds_per_rule = n_preds_per_rule

        # set the beta and dirichlet parameters for the convex program
        # this maintains the old behavior while letting ourselves force
        # the train labels to be used
        init_mthd = 'unlabeled' if n_max_labeled < 1 else 'labeled'
        ds_used = dataset_valid if n_max_labeled > 0 else dataset_train
        # if force_train_as_labeled_dataset:
        #     ds_used = dataset_train
        self._get_rule_class_freq_params(ds_used, method=init_mthd,\
                labeled_params_unif_prior=labeled_params_unif_prior,\
                labeled_params_scale=labeled_params_scale)

        # if n_max_labeled > 0:
        #     self._get_rule_class_freq_params(dataset_valid)
        # else:
        #     self._get_rule_class_freq_params(dataset_train, method='unlabeled')

        # get problem and solve for the mode in the Bayesian BF posterior
        self.mode_prob, post_mode_var = self._get_post_mode_program(L)
        # self.mode_prob_status = self.mode_prob.solve(solver='MOSEK', \
        #         verbose=True)
        self.mode_prob_status = self.mode_prob.solve(solver=self.solver, \
                verbose=self.verbose)

        # recover the posterior mode labeling
        self.post_mode_labeling = post_mode_var.value

        # compute the rule accuracies/class frequencies using the posterior
        # mode labeling so we can recover the BF weights
        # pm for posterior mode
        pm_class_freq_cts = np.sum(self.post_mode_labeling, axis=0)
        pm_rule_conf_mats = []
        # pm_rule_acc_cts = np.zeros(self.p)
        for j in range(self.p):
            pm_rule_conf_mats.append(self.post_mode_labeling.T @ L_aug[j])
            # pm_rule_acc_cts[j] = np.sum(L_aug[j] * self.post_mode_labeling)

        # get problem to solve for weights
        # self.wt_prob, sigma, gamma = self._make_dual_cp(L_aug, pm_rule_acc_cts,
        self.wt_prob, sigma, gamma = self._make_dual_cp(L_aug, pm_rule_conf_mats,
                pm_class_freq_cts)

        self.wt_prob.solve(solver='MOSEK', verbose=self.verbose)
        self.class_freq_wts = gamma.value
        if self.constraint_type == 'accuracy':
            self.param_wts = sigma.value
        else:
            self.param_wts = []
            for j in range(self.p):
                self.param_wts.append(sigma[j].value)


    def _get_post_mode_program(self, L):
        # for the posterior mode, we are solving the primal problem

        L_aug = self._initialize_L_aug(L)
        n_rule = len(L_aug)
        n_pts = L_aug[0].shape[0]
        n_class = L_aug[0].shape[1]

        z = cp.Variable((n_pts, n_class))
        # self.class_freq_param = cp.Parameter(n_class)
        # self.acc_param = cp.Parameter(n_rule)

        # class freq prior
        class_freq_prior = self.class_freq_params / self.class_freq_params.sum()

        constrs = [z >= 0, cp.sum(z, axis=1) == 1]

        # construct the objective in parts
        obj = - 1 * cp.sum(cp.entr(z))

        for l in range(n_class):
            if self.class_freq_params[l] > 1:
                obj -= (self.class_freq_params[l] - 1) * cp.log(cp.sum(z[:, l]))

        for j in range(self.p):
            conf_mat = z.T @ sp.sparse.csr_matrix(L_aug[j])
            if self.constraint_type == 'accuracy':
                conf_mat_tr = cp.trace(conf_mat)
                # the accuracy of rule j's prediction with z as the labeling
                rule_j_acc = conf_mat_tr / self.n_preds_per_rule[j]
                # if there is something with a coefficient of 0, the MOSEK does not
                # like it and the primal/dual solutions vary by a huge amount
                if self.rule_params[j][0] > 1:
                    obj -= (self.rule_params[j][0] - 1) * cp.log(rule_j_acc)
                if self.rule_params[j][1] > 1:
                    obj -= (self.rule_params[j][1] - 1) * cp.log(1 - rule_j_acc)

            # for these next two cases, we need to make the confusion matrix
            # row stochastic since we need k distributions (one per row).
            # however, doing this normalization would render the program
            # non-convex.  therefore, we divide by a constant like the above
            # case.  this constant will be determined by the prior class dist.
            elif self.constraint_type == 'class_accuracy':
                for l in range(self.n_class):
                    rule_j_class_l_acc = conf_mat[l, l] / (class_freq_prior[l] * self.n_preds_per_rule[j]) + 1e8
                    if self.rule_params[j][l][0] > 1:
                        obj -= (self.rule_params[j][l][0] - 1) * cp.log(rule_j_class_l_acc)
                    if self.rule_params[j][l][1] > 1:
                        obj -= (self.rule_params[j][l][1] - 1) * cp.log(1 - rule_j_class_l_acc)
            else:
                for l in range(self.n_class):
                    rule_j_class_l_acc = conf_mat[l, :] / class_freq_prior_cts[l]
                    for ll in range(self.n_class):
                        if self.rule_params[j][l][ll] > 1:
                            obj -= (self.rule_params[j][l][ll] - 1) * cp.log(rule_j_class_l_acc[ll])

        obj /= n_pts
        # also divide by the largest constant in front of logarithm to ensure
        # numerical stability -- i.e. so the objective doesn't blow up in size
        max_param = np.max(self.rule_params)
        max_rule_param_list = [np.max(ele) for ele in self.rule_params]
        max_param = max(max_param, np.max(max_rule_param_list))
        obj /= max_param
        # print(self.rule_params)
        # print(self.class_freq_params)
        objective = cp.Minimize(obj)
        return cp.Problem(objective, constrs), z

    def _make_dual_cp(self, L_aug, conf_mat_cts, class_freq_cts):
        # computes the weights such that the marginal accuracy of the resulting
        # prediction is equal to the computed expected labeling
        # make the dual convex program, which is solved in fit

        # create variables
        gamma = cp.Variable(self.n_class)
        if self.constraint_type == 'accuracy':
            sigma = cp.Variable(self.p)
        elif self.constraint_type == 'class_accuracy':
            sigma = [cp.Variable(self.n_class) for j in range(self.p)]
        else:
            sigma = [cp.Variable((self.n_class, self.n_class))
                    for j in range(self.p)]

        # create objective
        sigma_term = 0
        # sigma_abs_term = 0
        # for the class frequency terms
        gamma_term = gamma @ class_freq_cts

        # sigma_term = sigma @ acc_cts
        for j in range(self.p):
            if self.constraint_type == 'accuracy':
                sigma_term += sigma[j] * conf_mat_cts[j].trace()
            elif self.constraint_type == 'class_accuracy':
                sigma_term += sigma[j] @ np.diag(conf_mat_cts[j])
            else:
                sigma_term += cp.sum(cp.multiply(sigma[j], conf_mat_cts[j]))

        aggregated_weights = self._aggregate_weights(L_aug, sigma, gamma)

        obj = cp.Maximize(sigma_term
                + gamma_term
                - cp.sum(cp.log_sum_exp(aggregated_weights, axis = 1)))

        # create constraints
        constrs = []

        return cp.Problem(obj, constrs), sigma, gamma

    def _get_rule_class_freq_params(self, dataset, method='labeled',\
                labeled_params_scale=1, labeled_params_unif_prior=True):

        L = dataset[0]
        y = np.squeeze(dataset[1])

        if method == 'unlabeled':
            if self.unsup_balance_init == 'majority_vote':
                # use majority vote to estimate class freq parameters
                mv_label_model = MajorityVoting()
                mv_label_model.fit(dataset)
                mv_pred = mv_label_model.predict_proba(dataset)

                self.class_freq_params = mv_pred.sum(axis=0)

                # allow ourselves to pretend that there are a different amount
                # of ``labeled'' points when initializing the dirichlet prior
                if self.unsup_balance_init_rescale > 0:
                    self.class_freq_params *= self.unsup_balance_init_rescale \
                            / self.class_freq_params.sum()

                # assume a uniform prior, which is why we add stuff
                self.class_freq_params += 1
            elif self.unsup_balance_init == 'uniform':
                self.class_freq_params = np.ones(self.n_class)

            # now intialize the rule parameters
            if self.constraint_type == 'accuracy':
                # just assign (4, 1) as Beta parameters
                self.rule_params = [np.array([4, 1]) for _ in range(self.p)]
            elif self.constraint_type == 'class_accuracy':
                self.rule_params = [[np.array([4,1]) for _ in range(self.n_class)] for _ in range(self.p)]
            else:
                # a matrix with 4's on main diagonal and 1 everywhere else
                self.rule_params = [np.eye(self.n_class)*3 + 1\
                        for _ in range(self.p)]

            # record this stuff so saving the results doesn't break
            self.n_rules_used = self.p
            self.avg_labeled_per_rule = 0

        elif method == 'labeled':
            if self.constraint_type in ['class_accuracy', 'confusion_matrix']:
                raise ValueError("Labeled data estimation of parameters not"
                        f"implemented for {self.constraint_type} constraints")
            n_all_labeled = len(y)

            # sample the datapoints if need be
            if self.n_max_labeled > 0 and len(y) > self.n_max_labeled:
                choices = np.random.choice(np.arange(len(y)),self.n_max_labeled,
                        replace=False)
                y = y[choices]
                L = L[choices, :]
                dataset = [L, y]

            L_aug = self._initialize_L_aug(L)
            y_aug = self._initialize_one_hot_labels(y)
            p, n_labeled, _ = L_aug.shape

            # count the number of predictions made per rule
            n_rule_preds = np.sum(L_aug, axis=(-2, -1))

            # compute and report the number of classifiers with labeled data,
            # and average number of labeled data per classifier (ignoring
            # classifiers that got no labeled data)
            labeled_rules = n_rule_preds > 0
            n_labeled_rules = int(np.sum(labeled_rules))
            self.n_rules_used = n_labeled_rules
            self.avg_labeled_per_rule = np.mean(n_rule_preds[labeled_rules])
            self.logger.info('----Estimating Probabilities----')
            self.logger.info('%d out of %d labeled datapoints on average per rule ',
                    self.avg_labeled_per_rule, n_all_labeled)
            self.logger.info('%d rules with labeled data out of %d total',\
                    n_labeled_rules, p)
            self.logger.info('Number of labeled predictions per rule:'
                    f' {n_rule_preds[labeled_rules]}')
            if self.n_max_labeled > 0:
                self.logger.info('Using at maximum %d many labeled points',
                        self.n_max_labeled)

            # compute class distribution
            # need to use minlength argument because edge case is where a class
            # gets no labeled data
            class_freq_cts = np.bincount(y, minlength=self.n_class)
            class_freq_probs = class_freq_cts / n_labeled

            n_classes_with_labels = int(np.sum(class_freq_probs > 0))
            if n_classes_with_labels < self.n_class:
                self.logger.info('%d class(es) without labels out of %d total',\
                        n_classes_with_labels, self.n_class)

            # figure out the positions of each class
            # class_pos = np.full((n_labeled, self.n_class), False, dtype=bool)
            # for l in range(self.n_class):
            #     class_pos[:, l] = y == l

            offset = int(labeled_params_unif_prior)
            # so long as the value isn't 0 or -1
            if labeled_params_scale > 0:
                class_freq_cts = labeled_params_scale * class_freq_probs
            self.class_freq_params = class_freq_cts + offset

            rule_params = []
            for j in range(p):
                if n_rule_preds[j] == 0:
                    rule_params.append(np.ones(2))
                    continue

                # deal with abstentions by only looking at vector of labels
                # where a prediction was made on that point
                pred_on = L[:, j] != ABSTAIN

                n_corr_preds = (L[pred_on, j] == y[pred_on]).sum()
                # start with [1,1] as the beta parameters
                # if labeled_params_unif_prior is true. add to first position
                # for correct predictions, second position for incorrect ones
                jth_rule_params = np.array([n_corr_preds,\
                        n_rule_preds[j] - n_corr_preds])
                # print(jth_rule_params)
                if labeled_params_scale > 0:
                    jth_rule_params = jth_rule_params/jth_rule_params.sum()
                    jth_rule_params *= labeled_params_scale
                rule_params.append(jth_rule_params + offset)

            self.rule_params = rule_params
            # print(self.class_freq_params, self.rule_params)

    def _initialize_L_aug(self, L):
        # convert L into a stack of matrices, one-hot encodings of each
        # classifier's predictions. index 0 is which classifier, index 1 is
        # the datapoint index, index 2 is the class index
        L = L.T
        L_aug = (np.arange(self.n_class) == L[..., None]).astype(int)
        return L_aug

    def _initialize_one_hot_labels(self, y):
        # used to convert ground truth labels into one hot encoded labels.
        # rows are datapoints, columns are classes
        return np.squeeze(self._initialize_L_aug(y))

    def _aggregate_weights(self, L_aug, param_wts, class_freq_wts, mod=cp):
        # essentially create the weighted majority vote with provided weights

        # assuming param_wts is a k by k matrix where element ij is the weight
        # associated with the classifier predicting j when true label is i.
        p, n, _ = L_aug.shape
        n_class = self.n_class

        y_pred = mod.multiply(np.ones([n, n_class]), class_freq_wts[None])

        for j in range(p):
            # pick out column of confusion matrix (since we see observed pred)
            # for every datapoint.  Resulting matrix is n by k
            if self.constraint_type == 'accuracy':
                y_pred += mod.multiply(L_aug[j], param_wts[j])

            elif self.constraint_type == 'class_accuracy':
                y_pred += L_aug[j] @ mod.diag(param_wts[j])
            else:
            # for confusion matrices where param_wts is shape (p, k, k)
                y_pred += L_aug[j] @ param_wts[j].T
        return y_pred

    def _make_bf_preds(self, L_aug, param_wts, class_freq_wts):
        y_pred = self._aggregate_weights(L_aug, param_wts, class_freq_wts, mod=np)
        return sp.special.softmax(y_pred, axis=1)

    def predict_proba(self,
            dataset: Union[BaseDataset, np.ndarray],
            **kwargs: Any) -> np.ndarray:
            # weak: Optional[int] = None,
            # n_weaks: Optional[int] = None,
            # random_guess: Optional[int] = None,
            # seed: Optional[int] = None,
        # L = check_weak_labels(dataset, n_weaks=n_weaks, random_guess=random_guess)
        dataset_mod = dataset

        L = dataset_mod[0]

        L_aug = self._initialize_L_aug(L)
        y_pred = self._make_bf_preds(L_aug, self.param_wts, self.class_freq_wts)
        return y_pred

