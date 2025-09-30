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
    rand_methods = ["EBCC", "Snorkel", "MeTaL", "WeaSEL", "Denoise", "Fable", "CLL"]

    det_methods = [
        "MV",
        "DawidSkene",
        "HyperLM",
        "FlyingSquid",
        "IBCC",
        "BBF-U-Unif",
        "BBF-U-Emp",
        "BBF-U-EMP10",
    ]

    bf_folder = "BF/accuracy/"
    bbf_folder = "Bayesian_BF/accuracy/"
    bbf_fn_end = "_accuracy_semisup_100_validlabels.mat"
    bbf_unsup_fn_ends = [
        "_accuracy_unsup_trainlabels_unif.mat",
        "_accuracy_unsup_trainlabels_mv.mat",
        "_accuracy_unsup_trainlabels_mv_rescale10.mat",
    ]
    ebcc_fn_ends = ["_unif.mat", "_mv.mat", "_mv_rescale10.mat"]

    # load all the data from the saved files
    results_folder_path = "./results"
    losses = ["brier_score_train", "log_loss_train", "err_train", "f1_score_train"]

    n_methods = len(rand_methods) + len(det_methods)
    res_mdic = {}
    res_mdic["fit_times"] = np.zeros((n_methods, len(datasets)))
    for loss in losses:
        table_n_cols = len(datasets) if loss != "f1_score_train" else 7
        dist_results = np.zeros((n_methods, table_n_cols))
        agg_loss = np.zeros(dist_results.shape)
        res_mdic[loss + "rankings"] = np.zeros((n_methods, table_n_cols))

        bf_oracle_loss = np.zeros(table_n_cols)

        for d_ind, dataset in enumerate(datasets):
            if loss == "f1_score_train" and dataset in [
                "trec",
                "semeval",
                "chemprot",
                "agnews",
            ]:
                continue
            print(f"==========Dataset: {dataset}==========\n")
            dataset_result_path = os.path.join(results_folder_path, dataset + "_merged")

            name = []
            mdl_losses = []
            fit_times = []
            print(f"------Loss: {loss}")
            ### load all the results
            first_bbf_ind = 5
            for i, method in enumerate(det_methods):
                if i < first_bbf_ind:
                    method_result_path = os.path.join(dataset_result_path, method)
                    if i == 1:
                        method_result_path = os.path.join(method_result_path, "general")
                    method_fn = method + "_" + dataset + ".mat"
                else:
                    method_result_path = os.path.join(dataset_result_path, bbf_folder)
                    method_fn = (
                        "Bayesian_BF_" + dataset + bbf_unsup_fn_ends[i - first_bbf_ind]
                    )

                method_fn_full = os.path.join(method_result_path, method_fn)

                # load data
                mdic = sio.loadmat(method_fn_full)
                # in this case, only 1 loss since the model is deterministic
                mdl_loss = np.repeat(mdic[loss], 10)
                mdl_losses.append(mdl_loss)
                fit_times.append(mdic["fit_elapsed_time"])
                name.append(method)

            for i, method in enumerate(rand_methods):
                if i == 0:
                    method_result_path = os.path.join(dataset_result_path, "EBCC")
                    method_fn = "EBCC_" + dataset + ebcc_fn_ends[1]
                elif i < 7:
                    method_result_path = os.path.join(dataset_result_path, method)
                    method_fn = method + "_" + dataset + ".mat"

                method_fn_full = os.path.join(method_result_path, method_fn)

                # load data
                mdic = sio.loadmat(method_fn_full)
                mdl_losses.append(mdic[loss].squeeze())
                fit_times.append(mdic["fit_elapsed_time_mean"])

                name.append(method)

            # load BF oracle to aggregate its results
            method_result_path = os.path.join(dataset_result_path, bf_folder)
            method_fn = "BF_" + dataset + "_accuracy_semisup_trainlabels_eqconst.mat"
            bf_oracle_fn_full = os.path.join(method_result_path, method_fn)
            mdic = sio.loadmat(bf_oracle_fn_full)
            bf_oracle_loss[d_ind] = mdic[loss]

            # permute the methods so they're in the order we want

            # wrench dataset permutation
            # ['MV', 'DawidSkene', 'HyperLM', 'FlyingSquid', 'IBCC', 'BBF-U-Unif', 'BBF-U-Emp', 'BBF-U-EMP10', 'EBCC', 'Snorkel', 'MeTaL', 'WeaSEL', 'Denoise', 'Fable', CLL]
            # ['MV', 'DawidSkene', 'Snorkel', 'FlyingSquid', 'MeTaL', 'IBCC', 'EBCC', 'CLL', 'HyperLM', 'WeaSEL', 'Denoise', 'Fable', 'BBF-U-Unif', 'BBF-U-Emp', 'BBF-U-EMP10']

            start_perm = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
            end_perm = [8, 3, 5, 12, 13, 14, 6, 2, 4, 9, 10, 11, 7]
            name = swap_eles(name, start_perm, end_perm)
            mdl_losses = swap_eles(mdl_losses, start_perm, end_perm)
            fit_times = swap_eles(fit_times, start_perm, end_perm)

            # save the fit times
            for i in range(n_methods):
                res_mdic["fit_times"][:, d_ind] = fit_times

            mean_mdl_losses = []
            for mdl_loss in mdl_losses:
                mean_mdl_losses.append(np.round(np.mean(mdl_loss), 4))

            mean_mdl_losses = np.array(mean_mdl_losses)

            print(mean_mdl_losses)

            agg_loss[:, d_ind] = mean_mdl_losses

            flip = -1 if loss == "f1_score_train" else 1

            sorted_inds = np.argsort(flip * mean_mdl_losses)

            # also rank the losses here
            res_mdic[loss + "rankings"][:, d_ind] = rankdata(
                flip * mean_mdl_losses, method="min"
            )

            best_mthd_ind = sorted_inds[0]
            indist_inds = [best_mthd_ind]
            dist_results[best_mthd_ind, d_ind] = 1

            print(
                f"Best Method: {name[best_mthd_ind]} {mean_mdl_losses[best_mthd_ind]}"
            )
            # print(f'--Methods indistinguishable by 2 sided t-test p=0.05--')
            indist = []
            for i, mthd in enumerate(name):
                if i == best_mthd_ind:
                    continue

                # if identical arrays, we don't want to run a ttest
                res_norm = np.linalg.norm(mdl_losses[best_mthd_ind] - mdl_losses[i])
                if np.isclose(res_norm, 0):
                    p_val = 100
                else:
                    _, p_val = ttest_rel(mdl_losses[best_mthd_ind], mdl_losses[i])

                if p_val > 0.05:
                    indist.append(name[i] + f"({mean_mdl_losses[i]})")
                    dist_results[i, d_ind] = 1
                    indist_inds.append(i)

            if len(indist) > 0:
                print()
                print("Indistinguishable (p=0.05): ", end="")
                print(*indist, sep=", ")

            print()

            res_mdic[loss + "_ttest"] = dist_results
            res_mdic[loss] = agg_loss
            res_mdic["bf_oracle_" + loss] = bf_oracle_loss

        res_mdic["method_names"] = name
        res_mdic[loss + "avg_rank"] = res_mdic[loss + "rankings"].mean(axis=1)

    fn = "loss_labeled_ttest.mat"
    savemat(os.path.join(results_folder_path, fn), res_mdic)
