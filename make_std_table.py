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
    methods = ["Snorkel", "MeTaL", "EBCC", "WeaSEL", "Denoise", "Fable", "CLL"]

    # load all the data from the saved files
    results_folder_path = "./results"

    # bs_ll_table = Texttable()
    # err_table = Texttable()
    # bs_ll_table_rows = []

    # for rows in [bs_ll_table_rows, err_table_rows]:
    #     for method in methods:
    #         rows.append([method])

    fn = "loss_std.mat"

    mdic = sio.loadmat(os.path.join(results_folder_path, fn))
    all_std_vals = [
        mdic["brier_score_train_std"],
        mdic["log_loss_train_std"],
        mdic["err_train_std"],
        mdic["f1_score_train_std"],
    ]
    loss_names = ["brier_score", "log_loss", "err", "f1_score"]
    loss_names_nice = ["Brier Score", "Log Loss", "Accuracy", "F1 Score"]

    for i, loss in enumerate(loss_names):
        table = Texttable()
        table_rows = []
        for method in methods:
            table_rows.append([method])

        ### make classification error table
        for d_ind, dataset in enumerate(datasets):
            # skip the datasets where no f1 score is recorded
            if i == 3 and d_ind >= 7:
                continue
            # make loss table
            for j, row in enumerate(table_rows):
                scale = 1 if i == 1 else 100
                curr_val = scale * all_std_vals[i][j, d_ind]
                # if i == 2:
                # curr_val = scale - curr_val
                val = np.round(curr_val, 2)

                # need $ because for some reason latextable will always print 3
                # digits after the . even though the string is xx.xx
                app_value = "${value:.2f}$".format(value=val)
                row.append(app_value)

        datasets_used = datasets if i < 3 else datasets[:7]
        header = copy.deepcopy(datasets_used)
        header.insert(0, "Dataset")
        table_rows.insert(0, header)
        table.set_cols_align(["c"] * (1 + len(datasets_used)))
        table.add_rows(table_rows)
        caption = loss_names_nice[i]
        label = "tab:" + "wrench_" + loss + "_std"
        table_latex = latextable.draw_latex(
            table, use_booktabs=True, caption=caption, label=label
        )
        if i == 0:
            write_style = "w"
        else:
            write_style = "a"
        with open("results/" + "std_tables.txt", write_style) as f:
            f.write(table_latex)
            f.close()

    ### make a table combining f1 and accuracy, like in the FABLE paper
    loss = "f1_score"
    i = 3
    table = Texttable()
    table_rows = []
    for method in methods:
        table_rows.append([method])

    ### make classification error table
    # add row for BF oracle

    for d_ind, dataset in enumerate(datasets):
        # make 0-1 loss table
        for j, row in enumerate(table_rows):
            ref_loss_val = (
                all_std_vals[3][j, d_ind] if d_ind < 7 else all_std_vals[2][j, d_ind]
            )
            scale = 100
            curr_val = scale * ref_loss_val
            # if d_ind >= 7:
            # curr_val = scale - curr_val
            val = np.round(curr_val, 2)

            # need $ because for some reason latextable will always print 3
            # digits after the . even though the string is xx.xx
            row.append("${value:.2f}$".format(value=val))

    header = copy.deepcopy(datasets)
    header.insert(0, "Dataset")
    table_rows.insert(0, header)
    table.set_cols_align(["c"] * (1 + len(datasets)))
    table.add_rows(table_rows)
    caption = "Combined F1 Acc Table Std"
    label = "tab:" + "wrench_f1_acc_combined_std"
    table_latex = latextable.draw_latex(
        table, use_booktabs=True, caption=caption, label=label
    )
    if i == 0:
        write_style = "w"
    else:
        write_style = "a"
    with open("results/" + "combined_f1_acc_std_table.txt", write_style) as f:
        f.write(table_latex)
        f.close()
