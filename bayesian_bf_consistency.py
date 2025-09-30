import os
import json
import logging
from time import perf_counter

import numpy as np
import scipy as sp
from numpy.matlib import repmat
from scipy.io import savemat
import scipy.io as sio
import matplotlib.pyplot as plt

from wrench._logging import LoggingHandler
from wrench.labelmodel import Bayesian_BF
from sklearn.metrics import log_loss

def run_bayesian_bf(
        dataset_prefix,
        dataset_name=None,
        constraint_form='accuracy',
        labeled_set='train',
        n_max_labeled=-1,
        unsup_balance_init=None,
        unsup_balance_init_rescale=None,
        verbose=False,
        # verbose=True,
        solver='MOSEK',
        ):

    #### Load dataset
    dataset_path = os.path.join(dataset_prefix, dataset_name + merged_txt + '.mat')
    data = sio.loadmat(dataset_path)
    train_data = [data['train_pred'], data['train_labels']]
    n_train_points = train_data[0].shape[0]
    n_train_rules = train_data[0].shape[1]

    #### Run label model: Bayesian_BF
    label_model = Bayesian_BF(solver=solver, verbose=verbose)

    res = []

    for scale in scale_list:
        print('Running scale: ', scale)
        label_model.fit(
                train_data,
                train_data,
                constraint_type=constraint_form,
                n_max_labeled=n_max_labeled,
                unsup_balance_init=unsup_balance_init,
                unsup_balance_init_rescale=unsup_balance_init_rescale,
                # should be true to match theory since we require convergence
                # of alpha^\prime and beta^\prime and not alpha, beta
                labeled_params_unif_prior = True,
                labeled_params_scale = scale,
                )

        ### make predictions
        pred = label_model.predict_proba(train_data)
        res.append(pred)
        n_pts = pred.shape[0]

    return n_pts, res

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

    # whether to enlarge the text in the graph
    enlarge = False
    # enlarge = True

    # whether to replot the data or compute the data then plot
    replot = False
    # replot = True

    use_synthetic = False

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
        datasets += ['imdb', 'youtube', 'sms', 'cdr', 'yelp', 'commercial',\
            'tennis', 'trec', 'semeval', 'chemprot', 'agnews']

    # make result folder if it doesn't exist
    bbf_cons_result_path = os.path.join(results_folder_path, 'Bayesian_BF_Consistency')
    if not os.path.exists(bbf_cons_result_path):
        os.makedirs(bbf_cons_result_path)

    constraint_types = ['accuracy']

    scale_list = [10, 100, 1000, 10000, 100000]
    # scale_list = [10, 100]
    # scale_list = [10]

    print("List of hyperparameter scales: ", scale_list)
    for dataset in datasets:
        print(f"====Now Running on {dataset}====")
        # read the config file
        config_filename = os.path.join(dataset_prefix, dataset\
                + '_bayesian_bf_configs.json')
        with open(config_filename, 'r') as read_file:
            cfgs = json.load(read_file)

        # for constraint_type in constraint_types:
        constraint_type = constraint_types[0]

        if not replot:
            n_pts, res = run_bayesian_bf(
                    dataset_prefix,
                    dataset,
                    constraint_form=constraint_type,
                    n_max_labeled=200000, # something massive so all labels are used
                    )

            # now load g^* from BF
            oracle_result_fn = 'BF_' + dataset + '_accuracy_semisup_trainlabels'+\
                    '_eqconst.mat'
            dataset_result_path = os.path.join(results_folder_path, dataset + merged_txt)
            oracle_bf_full = os.path.join(dataset_result_path, 'BF/accuracy/' +\
                        oracle_result_fn)

            oracle_mdic = sio.loadmat(oracle_bf_full)
            oracle_pred = oracle_mdic['pred_train'].squeeze()

            # compute the KL divergence from oracle pred to bayesian bf pred
            kl_div_list = []
            for bbf_pred in res:
                entropy = np.sum(sp.stats.entropy(oracle_pred, axis=1))
                kl_div_list.append((-1 * np.sum(oracle_pred * np.log(bbf_pred)) - entropy)/n_pts)

            # save the results
            mdic = {"scale_list": scale_list, "kl_divergences": kl_div_list}
            savemat(os.path.join(bbf_cons_result_path, dataset + '_BBF_consistency.mat'), mdic)

        else:
            mdic = sio.loadmat(os.path.join(bbf_cons_result_path, dataset + '_BBF_consistency.mat'))
            scale_list = mdic["scale_list"].squeeze()
            kl_div_list = mdic["kl_divergences"].squeeze()

        print(f"----Now plotting----")
        # now graph and save the picture
        if enlarge:
            plt.rcParams.update({'font.size': 18})

        fig, ax = plt.subplots()
        # fig.dpi = 1200
        suffix = '_enlarged' if enlarge else ''
        plot_fn = os.path.join(bbf_cons_result_path, dataset + '_BBF_consistency' + suffix + '.pdf')
        ax.plot(scale_list, kl_div_list, '-', color='blue', label=r'$\frac{1}{N}d(\vec{g}^{*}, \vec{g}^{bbf})$')
        plt.xscale('log')
        plt.yscale('log')

        ax.legend()
        ax.set_ylabel('KL Divergence')
        ax.set_xlabel(r'Hyperparameter Scale ($s$)')

        fig.savefig(plot_fn, bbox_inches='tight', format='pdf')
        plt.close(fig)

