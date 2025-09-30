import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.io import savemat
from scipy.stats import rankdata
from scipy.stats.mstats import ttest_rel


def swap_eles(lst, start, end):
    nlst = copy.deepcopy(lst)
    for s_ind, e_ind in zip(start, end):
        nlst[e_ind] = lst[s_ind]

    return nlst


if __name__ == "__main__":
    datasets = [
        "imdb",
        "youtube",
        "sms",
        "cdr",
        "yelp",
        "commercial",
        "tennis",
        "trec",
        "semeval",
        "chemprot",
        "agnews",
    ]
    rand_methods = ["Snorkel", "MeTaL", "EBCC", "WeaSEL", "Denoise", "Fable", "CLL"]

    ebcc_fn_ends = ["_unif.mat", "_mv.mat", "_mv_rescale10.mat"]

    # load all the data from the saved files
    results_folder_path = "./results"
    losses = ["brier_score_train", "log_loss_train", "err_train", "f1_score_train"]

    n_methods = len(rand_methods)
    res_mdic = {}
    for loss in losses:
        table_n_cols = len(datasets) if loss != "f1_score_train" else 7
        dist_results = np.zeros((n_methods, table_n_cols))
        agg_std = np.zeros(dist_results.shape)

        for d_ind, dataset in enumerate(datasets):
            if loss == "f1_score_train" and dataset in [
                "trec",
                "semeval",
                "chemprot",
                "agnews",
            ]:
                continue
            dataset_result_path = os.path.join(results_folder_path, dataset + "_merged")

            name = []
            mdl_losses = []
            fit_times = []
            ### load all the results
            for i, method in enumerate(rand_methods):
                if i == 2:
                    method_result_path = os.path.join(dataset_result_path, "EBCC")
                    method_fn = "EBCC_" + dataset + ebcc_fn_ends[1]
                elif i < 7:
                    method_result_path = os.path.join(dataset_result_path, method)
                    method_fn = method + "_" + dataset + ".mat"

                method_fn_full = os.path.join(method_result_path, method_fn)

                # load data
                mdic = sio.loadmat(method_fn_full)

                name.append(method)

                agg_std[i, d_ind] = mdic[loss + "_std"].squeeze()

            res_mdic[loss + "_std"] = agg_std

        res_mdic["method_names"] = name

    fn = "loss_std.mat"
    savemat(os.path.join(results_folder_path, fn), res_mdic)
