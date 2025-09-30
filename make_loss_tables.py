import os
import copy
import numpy as np
import scipy.io as sio
import latextable
from texttable import Texttable


def add_bold(val):
    return "\\textbf{{{value:.2f}}}".format(value=val)


if __name__ == "__main__":
    print(
        "Make sure you are out of the `wrench` environment and "
        "latextable=1.0.0 is installed."
    )

    # Cancer is Breast Cancer
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
    methods = [
        "MV",
        "DawidSkene",
        "Snorkel",
        "FlyingSquid",
        "MeTaL",
        "IBCC",
        "EBCC",
        "CLL",
        "HyperLM",
        "WeaSEL",
        "Denoise",
        "Fable",
        "BBF-U-Unif",
        "BBF-U-Emp-NOT-USED",
        "BBF-U-EMP10-NOT-USED",
    ]

    # load all the data from the saved files
    results_folder_path = "./results"

    # bs_ll_table = Texttable()
    # err_table = Texttable()
    # bs_ll_table_rows = []

    # for rows in [bs_ll_table_rows, err_table_rows]:
    #     for method in methods:
    #         rows.append([method])

    fn = "loss_labeled_ttest.mat"

    mdic = sio.loadmat(os.path.join(results_folder_path, fn))
    loss_vals = [
        mdic["brier_score_train"],
        mdic["log_loss_train"],
        mdic["err_train"],
        mdic["f1_score_train"],
    ]
    loss_bold = [
        mdic["brier_score_train_ttest"],
        mdic["log_loss_train_ttest"],
        mdic["err_train_ttest"],
        mdic["f1_score_train_ttest"],
    ]
    loss_names = ["brier_score", "log_loss", "err", "f1_score"]
    loss_names_nice = ["Brier Score", "Log Loss", "Accuracy", "F1 Score"]

    for i, loss in enumerate(loss_names):
        table = Texttable()
        table_rows = []
        for method in methods:
            table_rows.append([method])
        table_rows.append(["$g^{*}$"])

        ### make classification error table
        # add row for BF oracle
        all_loss_vals = np.vstack(
            (loss_vals[i], mdic["bf_oracle_" + loss + "_train"].squeeze())
        )
        all_loss_bold = np.vstack((loss_bold[i], np.zeros(loss_bold[i].shape[1])))

        for d_ind, dataset in enumerate(datasets):
            # skip the datasets where no f1 score is recorded
            if i == 3 and d_ind >= 7:
                continue
            # make 0-1 loss table
            for j, row in enumerate(table_rows):
                scale = 1 if i == 1 else 100
                curr_val = scale * all_loss_vals[j, d_ind]
                if i == 2:
                    curr_val = scale - curr_val
                val = np.round(curr_val, 2)

                # need $ because for some reason latextable will always print 3
                # digits after the . even though the string is xx.xx
                app_value = (
                    add_bold(val)
                    if all_loss_bold[j, d_ind]
                    else "${value:.2f}$".format(value=val)
                )
                row.append(app_value)

        datasets_used = datasets if i < 3 else datasets[:7]
        header = copy.deepcopy(datasets_used)
        header.insert(0, "Dataset")
        table_rows.insert(0, header)
        table.set_cols_align(["c"] * (1 + len(datasets_used)))
        table.add_rows(table_rows)
        caption = loss_names_nice[i]
        label = "tab:" + "wrench_" + loss
        table_latex = latextable.draw_latex(
            table, use_booktabs=True, caption=caption, label=label
        )
        if i == 0:
            write_style = "w"
        else:
            write_style = "a"
        with open("results/" + "error_tables.txt", write_style) as f:
            f.write(table_latex)
            f.close()

    ### make a table combining f1 and accuracy, like in the FABLE paper
    loss = "f1_score"
    i = 3
    table = Texttable()
    table_rows = []
    for method in methods:
        table_rows.append([method])
    table_rows.append(["$g^{*}$"])

    ### make classification error table
    # add row for BF oracle
    all_loss_vals = np.vstack(
        (loss_vals[i], mdic["bf_oracle_" + loss + "_train"].squeeze())
    )
    all_loss_bold = np.vstack((loss_bold[i], np.zeros(loss_bold[i].shape[1])))
    zo_loss_vals = np.vstack(
        (loss_vals[2], mdic["bf_oracle_" + loss_names[2] + "_train"].squeeze())
    )
    zo_loss_bold = np.vstack((loss_bold[2], np.zeros(loss_bold[2].shape[1])))

    for d_ind, dataset in enumerate(datasets):
        # make 0-1 loss table
        for j, row in enumerate(table_rows):
            ref_loss_val = (
                all_loss_vals[j, d_ind] if d_ind < 7 else zo_loss_vals[j, d_ind]
            )
            scale = 100
            curr_val = scale * ref_loss_val
            if d_ind >= 7:
                curr_val = scale - curr_val
            val = np.round(curr_val, 2)

            # need $ because for some reason latextable will always print 3
            # digits after the . even though the string is xx.xx
            bold_ref_val = (
                all_loss_bold[j, d_ind] if d_ind < 7 else zo_loss_bold[j, d_ind]
            )
            app_value = (
                add_bold(val) if bold_ref_val else "${value:.2f}$".format(value=val)
            )
            row.append(app_value)

    header = copy.deepcopy(datasets)
    header.insert(0, "Dataset")
    table_rows.insert(0, header)
    table.set_cols_align(["c"] * (1 + len(datasets)))
    table.add_rows(table_rows)
    caption = "Combined F1 Acc Table"
    label = "tab:" + "wrench_f1_acc_combined"
    table_latex = latextable.draw_latex(
        table, use_booktabs=True, caption=caption, label=label
    )
    if i == 0:
        write_style = "w"
    else:
        write_style = "a"
    with open("results/" + "combined_f1_acc_table.txt", write_style) as f:
        f.write(table_latex)
        f.close()

    ### now make a table for time to fit
    fit_time_table = Texttable()

    # get data
    ft = mdic["fit_times"]

    # make header
    fit_time_header = copy.deepcopy(datasets)
    fit_time_header.insert(0, "Dataset")

    # populate the rows
    fit_time_table_rows = [fit_time_header]
    for i, method in enumerate(methods):
        ft_row = [method]
        for j in range(len(datasets)):
            val = "${value:.2f}$".format(value=ft[i, j])
            ft_row.append(val)
        fit_time_table_rows.append(ft_row)

    fit_time_table.set_cols_align(["c"] * (1 + len(datasets)))
    fit_time_table.add_rows(fit_time_table_rows)
    fit_time_caption = "Average fit times (s)"
    fit_time_label = "tab:fit_time_stats"
    fit_time_table_latex = latextable.draw_latex(
        fit_time_table,
        use_booktabs=True,
        caption=fit_time_caption,
        label=fit_time_label,
    )
    with open("results/" + "fit_time.txt", "w") as f:
        f.write(fit_time_table_latex)
        f.close()
    # make rows
