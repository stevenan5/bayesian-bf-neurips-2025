import os
import json
import logging
from time import perf_counter

import numpy as np
from numpy.matlib import repmat
from scipy.io import savemat
import scipy.io as sio
import matplotlib.pyplot as plt

from wrench._logging import LoggingHandler
from wrench.labelmodel import Bayesian_BF
from sklearn.metrics import log_loss

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

def run_bayesian_bf(
        dataset_prefix,
        dataset_name=None,
        n_classes=0,
        constraint_form='accuracy',
        labeled_set='valid',
        n_max_labeled=-1,
        unsup_balance_init='majority_vote',
        unsup_balance_init_rescale=10,
        n_runs = 1,
        replot=False,
        use_test=True,
        verbose=False,
        # verbose=True,
        save_path=None,
        solver='MOSEK',
        logger=logger,
        ):

    #### Load dataset
    dataset_path = os.path.join(dataset_prefix, dataset_name + merged_txt + '.mat')
    data = sio.loadmat(dataset_path)
    train_data = [data['train_pred'], data['train_labels']]
    n_train_points = train_data[0].shape[0]
    n_train_rules = train_data[0].shape[1]

    # result path
    result_filename = get_result_filename(dataset, constraint_form,
            labeled_set, n_labeled=n_max_labeled,
            balance_init=unsup_balance_init,
            balance_init_rescale=unsup_balance_init_rescale)

    use_all_valid = False
    try:
        valid_data = [data['val_pred'], data['validation_labels']]
        use_all_valid = labeled_set == 'valid' and\
                (n_max_labeled < 0 or n_max_labeled == valid_data[0].shape[0])
    except KeyError:
        print('No validation set found!')

    if use_test:
        test_data = [data['test_pred'], data['test_labels']]

    if labeled_set == 'valid':
        labeled_data = valid_data
    else:
        labeled_data = train_data

    ### if we use all validation data/or are unsup, always force n_runs to be 1
    if use_all_valid or labeled_set == 'train':
        n_runs = 1

    #### Run label model: Bayesian_BF
    label_model = Bayesian_BF(solver=solver, verbose=verbose)

    for run_no in range(n_runs):
        if not replot:
            if n_runs > 1:
                logger.info('============Run Number %d============', run_no + 1)
            start_time = perf_counter()
            label_model.fit(
                    train_data,
                    labeled_data,
                    constraint_type=constraint_form,
                    n_max_labeled=n_max_labeled,
                    unsup_balance_init=unsup_balance_init,
                    unsup_balance_init_rescale=unsup_balance_init_rescale,
                    logger=logger,
                    )
            end_time = perf_counter()

            elapsed_time = end_time - start_time

            ### make predictions
            Y_p_train = label_model.predict_proba(train_data)
            # pred_train = np.argmax(Y_p_train, axis=1)
            true_labels_train = np.squeeze(train_data[1])
            if use_test:
                Y_p_test = label_model.predict_proba(test_data)
                # pred_test = np.argmax(Y_p_test, axis=1)
                true_labels_test = np.squeeze(test_data[1])

            ### compute losses
            brier_score_train = multi_brier(true_labels_train, Y_p_train)
            logloss_train = log_loss(true_labels_train,Y_p_train)
            acc_train = label_model.test(train_data, 'acc')
            if n_class == 2:
                f1_score_train = label_model.test(train_data, 'f1_binary')

            if run_no == 0:
                mdic = {
                        # "pred_train": [],
                        "log_loss_train": [],
                        # "true_labels_train": true_labels_train,
                        "brier_score_train": [],
                        "acc_train": [],
                        "err_train": [],
                        "n_rules_used": [],
                        "rule_weights": [],
                        "class_freq_weights": [],
                        "avg_num_labeled_per_rule": [],
                        "fit_elapsed_time": [],
                        }
                if n_class == 2:
                    mdic["f1_score_train"] = []

            # mdic["pred_train"].append(Y_p_train)
            mdic["log_loss_train"].append(logloss_train)
            mdic["brier_score_train"].append(brier_score_train)
            if n_class == 2:
                mdic["f1_score_train"].append(f1_score_train)
            mdic["acc_train"].append(acc_train)
            mdic["err_train"].append(1 - acc_train)
            mdic["n_rules_used"].append(label_model.n_rules_used)
            mdic["rule_weights"].append(label_model.param_wts)
            mdic["class_freq_weights"].append(label_model.class_freq_wts)
            mdic["avg_num_labeled_per_rule"].append(\
                    label_model.avg_labeled_per_rule)
            mdic["fit_elapsed_time"].append(elapsed_time)
            # optimization problem finds the negative entropy and is not
            # averaged over the total number of datapoints

            if use_test:
                brier_score_test = multi_brier(true_labels_test, Y_p_test)
                logloss_test = log_loss(true_labels_test, Y_p_test)
                acc_test = label_model.test(test_data, 'acc')
                f1_score_test = label_model.test(test_data, 'f1_binary')
                if run_no == 0:
                    mdic_test = {
                                # "pred_test": [],
                                # "true_labels_test": true_labels_test,
                                "log_loss_test": [],
                                "brier_score_test": [],
                                "acc_test": [],
                                "err_test": [],
                                }

                    if n_class == 2:
                        mdic_test["f1_score_test"] = []

                    mdic.update(mdic_test)

                # mdic["pred_test"].append(Y_p_test)
                mdic["log_loss_test"].append(logloss_test)
                mdic["brier_score_test"].append(brier_score_test)
                mdic["acc_test"].append(acc_test)
                mdic["err_test"].append(1 - acc_test)

            ### report results
            logger.info('----------------Results----------------')
            logger.info('time to fit: %.1f seconds', elapsed_time)
            logger.info('train acc (train err): %.4f (%.4f)',
                    acc_train, 1 - acc_train)
            logger.info('train log loss: %.4f', logloss_train)
            logger.info('train brier score: %.4f', brier_score_train)
            if n_class == 2:
                logger.info('train f1 score: %.4f', f1_score_train)
            if use_test:
                logger.info('test acc (test err): %.4f (%.4f)',
                        acc_test, 1 - acc_test)
                logger.info('test log loss: %.4f', logloss_test)
                logger.info('test brier score: %.4f', brier_score_test)
        else:
            mdic = sio.loadmat(os.path.join(save_path, result_filename))

    # if number of runs is >1, report and store mean results and standard
    # deviations
    if n_runs > 1:
        mdic["log_loss_train_mean"]     = np.mean(mdic["log_loss_train"])
        mdic["brier_score_train_mean"]  = np.mean(mdic["brier_score_train"])
        mdic["acc_train_mean"]          = np.mean(mdic["acc_train"])
        if n_class == 2:
            mdic["f1_score_train_mean"]       = np.mean(mdic["f1_score_train"])
        mdic["err_train_mean"]          = np.mean(mdic["err_train"])
        mdic["n_rules_used_mean"]       = np.mean(mdic["n_rules_used"])
        mdic["avg_num_labeled_per_rule_mean"] =\
                np.mean(mdic["avg_num_labeled_per_rule"])
        mdic["fit_elapsed_time_mean"]   = np.mean(mdic["fit_elapsed_time"])

        mdic["log_loss_train_std"]     = np.std(mdic["log_loss_train"])
        mdic["brier_score_train_std"]  = np.std(mdic["brier_score_train"])
        mdic["acc_train_std"]          = np.std(mdic["acc_train"])
        mdic["err_train_std"]          = np.std(mdic["err_train"])
        if n_class == 2:
            mdic["f1_score_train_std"] = np.std(mdic["f1_score_train"])
        mdic["avg_num_labeled_per_rule_std"] =\
                np.std(mdic["avg_num_labeled_per_rule"])
        mdic["fit_elapsed_time_std"]   = np.std(mdic["fit_elapsed_time"])

        if use_test:
            mdic["log_loss_test_mean"]    = np.mean(mdic["log_loss_test"])
            mdic["brier_score_test_mean"] = np.mean(mdic["brier_score_test"])
            mdic["acc_test_mean"]         = np.mean(mdic["acc_test"])
            if n_class == 2:
                mdic["f1_score_test_mean"]= np.mean(mdic["f1_score_test"])
            mdic["err_test_mean"]         = np.mean(mdic["err_test"])

            mdic["log_loss_test_std"]    = np.std(mdic["log_loss_test"])
            mdic["brier_score_test_std"] = np.std(mdic["brier_score_test"])
            mdic["acc_test_std"]         = np.std(mdic["acc_test"])
            mdic["err_test_std"]         = np.std(mdic["err_test"])
            if n_class == 2:
                mdic["f1_score_test_std"]= np.std(mdic["f1_score_test"])

        logger.info('================Aggregated Results================')
        logger.info('Total number of runs: %d', n_runs)
        logger.info('Average time to fit: %.1f seconds (std: %.1f)',
                mdic['fit_elapsed_time_mean'], mdic['fit_elapsed_time_std'])
        logger.info('Average of %.2f rules out of %d had labeled data',
                mdic["n_rules_used_mean"], n_train_rules)
        logger.info('Average of %d labeled datapoints out of %d per rule (std: %.4f)',
                mdic["avg_num_labeled_per_rule_mean"], n_max_labeled,\
                        mdic["avg_num_labeled_per_rule_std"])
        logger.info('train mean acc +- std (mean train err):'
                ' %.4f +- %.4f (%.4f)', mdic['acc_train_mean'],
                mdic['acc_train_std'], mdic['err_train_mean'])
        logger.info('train mean log loss +- std: %.4f +- %.4f',
                mdic['log_loss_train_mean'], mdic['log_loss_train_std'])
        logger.info('train mean brier score +- std: %.4f +- %.4f',
                mdic['brier_score_train_mean'], mdic['brier_score_train_std'])
        if n_class == 2:
            logger.info('train mean f1 score +- std: %.4f +- %.4f',
                    mdic['f1_score_train_mean'], mdic['f1_score_train_std'])

        if use_test:
            logger.info('test mean acc +- std (mean test err):'
                    ' %.4f +- %.4f (%.4f)', mdic['acc_test_mean'],
                    mdic['acc_test_std'], mdic['err_test_mean'])
            logger.info('test mean log loss +- std: %.4f +- %.4f',
                    mdic['log_loss_test_mean'], mdic['log_loss_test_std'])
            logger.info('test mean brier score +- std: %.4f +- %.4f',
                    mdic['brier_score_test_mean'], mdic['brier_score_test_std'])
            if n_class == 2:
                logger.info('test mean f1 score +- std: %.4f +- %.4f',
                        mdic['f1_score_test_mean'], mdic['f1_score_test_std'])

    if not replot:
        savemat(os.path.join(save_path, result_filename), mdic)

    return mdic

def get_result_filename(dataset_name, constraint_name,
        labeled_set, n_labeled=None, balance_init=None,
            balance_init_rescale=None):

    unsup_balance_init = ''
    unsup_balance_init_rescale = ''

    if labeled_set == 'valid':
        labeled_set_used = 'validlabels'
        est_pdgm = 'semisup'
    elif labeled_set == 'train':
        labeled_set_used = 'trainlabels'
        est_pdgm = 'unsup'
        if balance_init == 'majority_vote':
            unsup_balance_init = '_mv'
        elif balance_init == 'uniform':
            unsup_balance_init = '_unif'
        if balance_init_rescale > 0:
            unsup_balance_init_rescale = '_rescale' + str(balance_init_rescale)


    n_lab = ''
    if n_labeled is not None:
        if n_labeled > 0  and labeled_set == 'valid':
            n_lab = str(n_labeled) + '_'

    filename = "Bayesian_BF_"\
            + dataset_name + '_'\
            + constraint_name + '_'\
            + est_pdgm + '_'\
            + n_lab\
            + labeled_set_used\
            + unsup_balance_init\
            + unsup_balance_init_rescale\
            + ".mat"

    return filename

def multi_brier(labels, pred_probs):
    """
    multiclass brier score
    assumes labels is 1d vector with elements in {0, 1, ..., n_class - 1}
    position of the ture class and 0 otherwise
    """
    n_class = int(np.max(labels) + 1)
    labels = (np.arange(n_class) == labels[..., None]).astype(int)
    sq_diff = np.square(labels - pred_probs)
    datapoint_loss = np.sum(sq_diff, axis=1)

    return np.mean(datapoint_loss)

# pylint: disable=C0103
if __name__ == '__main__':
    # we only want to test certain combinations
    # semi-supervised, validation set, use bounds
    # unsupervised, training set, use bounds (akin to crowdsourcing)
    # oracle, training set, no bounds (equality constraints)

    # create results folder if it doesn't exist
    results_folder_path = './results'
    if not os.path.exists(results_folder_path):
        os.makedirs(results_folder_path)

    # path for config jsons
    dataset_prefix = './datasets/'

    # whether or not to use the version of the datasets where the
    # train/test/valid sets are merged
    use_merged = True
    # use_merged = False

    merged_txt = '_merged' if use_merged else ''

    # whether or not bayesian bf should be run on synthetic datasets
    use_synthetic = False
    # use_synthetic = True

    # whether or not figures are to be replot from saved data
    replot_figs = False
    # replot_figs = True

    datasets = []
    if use_synthetic:
        # change dataset path if using synthetic datasets
        dataset_prefix = os.path.join(dataset_prefix, 'synthetic')
        # create dataset names for synthetic datasets
        syth_filename_part = 'synth_10p_1000n_100nval__'
        n_synth = 10
        for i in range(n_synth):
            datasets.append(syth_filename_part + str(i))
    else:
        # wrench datasets
        # datasets += ['aa2', 'basketball', 'breast_cancer', 'cardio', 'domain',\
               # 'imdb', 'obs', 'sms', 'yelp', 'youtube']
        datasets = ['imdb', 'youtube', 'sms', 'cdr', 'yelp', 'commercial',\
            'tennis', 'trec', 'semeval', 'chemprot', 'agnews']
        # crowdsourcing datasets
        # datasets += ['bird', 'rte', 'dog', 'web']

    datasets = ['imdb']

    constraint_types = ['accuracy']

    for dataset in datasets:
        # read the config file
        config_filename = os.path.join(dataset_prefix, dataset\
                + '_bayesian_bf_configs.json')
        with open(config_filename, 'r') as read_file:
            cfgs = json.load(read_file)

        # make result folder if it doesn't exist
        dataset_result_path = os.path.join(results_folder_path, dataset + merged_txt)
        if not os.path.exists(dataset_result_path):
            os.makedirs(dataset_result_path)
        # make folder for Bayes_BF specifically
        method_result_path = os.path.join(dataset_result_path, 'Bayesian_BF')
        if not os.path.exists(method_result_path):
            os.makedirs(method_result_path)

        for cfg in cfgs:
            n_class = cfg['n_classes']
            # get list of labeled datapoint counts to run and deleted it
            n_max_labeled_list = cfg['n_max_labeled']
            del cfg['n_max_labeled']

            # for constraint_type in constraint_types:
            constraint_type = constraint_types[0]
            cons_result_path = os.path.join(
                    method_result_path, constraint_type)

            if not os.path.exists(cons_result_path):
                os.makedirs(cons_result_path)

            for n_labeled in n_max_labeled_list:
                if not replot_figs:
                    # change loggers every time we change settings
                    # remove old handlers
                    for handler in logger.handlers[:]:
                        logger.removeHandler(handler)
                    formatter=logging.Formatter('%(asctime)s - %(message)s',
                            '%Y-%m-%d %H:%M:%S')
                    log_filename = get_result_filename(
                            dataset,
                            constraint_type,
                            cfg['labeled_set'],
                            n_labeled=n_labeled,
                            balance_init=cfg['unsup_balance_init'],
                            balance_init_rescale=cfg['unsup_balance_init_rescale'],
                            )[:-4] + '.log'
                    log_filename_full = os.path.join(cons_result_path,
                            log_filename)
                    file_handler=logging.FileHandler(log_filename_full, 'w')
                    file_handler.setFormatter(formatter)
                    logger.addHandler(file_handler)
                    # log all the run parameters
                    logger.info('==========Running New Instance==========')
                    logger.info('dataset: %s, n_class: %d', dataset,n_class)
                    logger.info('constraint type: %s, labeled set: %s',
                            constraint_type, cfg['labeled_set'])
                    if cfg['labeled_set'] == 'valid':
                        logger.info('n_max_labeled: %s', n_labeled)
                    elif cfg['labeled_set'] == 'train':
                        logger.info('balance init: %s', cfg['unsup_balance_init'])
                        if cfg['unsup_balance_init_rescale'] > 0:
                            logger.info('balance init rescale: %s',
                                    cfg['unsup_balance_init_rescale'])

                run_bayesian_bf(
                        dataset_prefix,
                        constraint_form=constraint_type,
                        save_path=cons_result_path,
                        replot=replot_figs,
                        logger=logger,
                        n_max_labeled=n_labeled,
                        **cfg
                        )
