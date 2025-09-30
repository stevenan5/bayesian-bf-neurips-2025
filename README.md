# bayesian-bf-neurips-2025

This repo contains code for Statistical Analysis of an Adversarial Bayesian Weak Supervision Method

If you use this in an academic study, please cite the paper:
```
coming soon!
```

This is a modification of the code provided for "Convergence Behavior of an Adversarial Weak Supervision Method" found [here](https://github.com/stevenan5/balsubramani-freund-uai-2024).
We have added our implementation of Bayesian BF.
This README is an edited version of the one provided in the above repository.
The results seen in the paper have been provided, but we also describe how to run the methods for yourself.

Scripts have been written to automatically run and record the results (0-1 Loss, Brier Score, Log Loss) of multiple methods on select datasets (more than once if the method has randomness).

Finally, code that creates a latex table based on the results of the methods on the datasets has also been included.

More detailed explanation is given below, along with installation instructions.

## Installation
1. create and activate conda environment using the `environment_bayesian_bf.yml` file and NOT the `environment.yml` file

    `conda env create -f environment_bayesian_bf`

2. activate the environment

    `conda activate wrench_bayesian_bf`
    
## Datasets
Three of the methods we test need to have the dataset features.
This is done by downloading the files from Hugging Face at [jieyuz2/WRENCH](https://huggingface.co/datasets/jieyuz2/WRENCH/tree/main/classification).
The files will need to be placed within the folder `datasets/datasets_with_features`.
E.g. for `cdr`, the path for files should look like `datasets/datasets_with_features/cdr/train.json`.

## Running the methods
We have provided files `python3 run_(mv|ds|snorkel|flyingsquid|metal|ibcc|ebcc|hyperlm|weasel|denoise|fable|bayesian_bf|cll).py` that will automatically run the named method on the following datasets (see `README_wrench.md` for links/sources):

- IMDB
- Youtube
- SMS
- CDR
- Yelp
- Commercial
- Tennis
- TREC
- SemEval
- ChemProt
- AG News

The 0-1 Loss, Brier Score, and Log Loss are all recorded, along with other information in the `*.mat` file.
A `results` folder is automatically created along with a folder for each dataset.
For each dataset, a folder will be created for each method.
That is where the `*.log` and `*.mat` files can be found.
Below is extra information about changeable settings/other instructions to run the method.

### Bayesian BF
To run BBF, ensure that the hyperparameter settings are written, then run BBF as follows.

1. Write the settings for all datasets.

    `python3 write_bayesian_bf_settings.py`
2. Run BF.

    `python3 run_bayesian_bf.py`
    
Note that the results in the paper are from `BBF Unif`.

### BF

BF relies on many hyperparameters, which have already been set in `write_bf_settings.py`.
The hyperparameters have been set so that g^* is computed when BF is run.

1. Write the settings for all datasets.

    `python3 write_bf_settings.py`
2. Run BF.

    `python3 run_bf.py`

### MV, Dawid-Skene, Snorkel (DP), FlyingSquid, MeTaL, iBCC, EBCC, HyperLM, WeaSEL, Denoise, FABLE, CLL
Running the respective `python3 run_(mv|ds|snorkel|flyingsquid|metal|ibcc|ebcc|hyperlm|weasel|denoise|fable|cll).py` suffices.

### Bayesian BF Consistency Graphs
This code will run BBF and generate graphs showing its consistency as described in the paper.
Simply run `python3 bayesian_bf_consistency.py`.

## Table Generation
We have also included code that automatically generates latex tables containing all results (in `results` folder).

- Result t-test/aggregation and std deviation retrieval

    Performs a two sided t-test on the error rates (each of the three losses).
    Also aggregates data from results for each dataset. 
    This must be run before making the loss tables (below).

    `python3 result_t_test.py`

    `python3 loss_std_retrieval.py`

- Table Generation

  Lastly, we provide a script to generate latex tables of F1/0-1 loss, 0-1 loss by itself, and the standard deviations of the F1/0-1 loss table.
  Note that you must use latextable version 1.0.0 to get multicolumn tables.
  However, Python 3.6 (which is required for the `wrench` package) is too old for that version, so one must use a newer version of Python, 3.9+ should work.

    `conda deactivate wrench`

    `pip install latextable==1.0.0`

    `python3 make_loss_tables.py`
    
    `python3 make_std_table.py`


## Miscellaneous Information
We store a simplified version of `wrench` datasets as `.mat` files.
They contain train, validation, and possibly test labels and labeling function predictions.
