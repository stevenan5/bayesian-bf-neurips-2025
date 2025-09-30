import os
import json
from copy import deepcopy

if __name__ == '__main__':
    prefix= './datasets'

    # store relevant kwarg combinations in dictionaries
    unsup_uniform_kwargs = {
            'labeled_set' : 'train',
            'n_max_labeled': [-1],
            'unsup_balance_init': 'uniform',
            'unsup_balance_init_rescale': -1,
            }

    unsup_mv_kwargs = {
            'labeled_set' : 'train',
            'n_max_labeled':[-1],
            'unsup_balance_init': 'majority_vote',
            'unsup_balance_init_rescale': -1,
            }

    unsup_mv_rescale_kwargs = {
            'labeled_set' : 'train',
            'n_max_labeled':[-1],
            'unsup_balance_init': 'majority_vote',
            'unsup_balance_init_rescale': 10,
            }

    semisup_kwargs = {
            'labeled_set' : 'valid',
            'n_max_labeled' : [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'unsup_balance_init': None,
            'unsup_balance_init_rescale': None,
            }

    # list out which general dictionaries to use
    # dics_to_use = [unsup_uniform_kwargs, unsup_mv_kwargs
    #         , unsup_mv_rescale_kwargs, semisup_kwargs]
    # dics_to_use = [unsup_uniform_kwargs, unsup_mv_kwargs
    #         , unsup_mv_rescale_kwargs]
    dics_to_use = [unsup_uniform_kwargs]
    # dics_to_use = [semisup_kwargs]

    ### wrench datasets
    aa2_dic = {
            'dataset_name': 'aa2',
            'n_classes': 2,
            'use_test' : False,
            # there are specific situations where this is ignored, specifically
            # when oracle accuracies are used, stuff is getting replot, or all
            # validation data is used.
            'n_runs' : 10,
            }
    # we want dataset specific entries to overwrite the general entries
    aa2_dics = [dict(gen_dic, **aa2_dic) for gen_dic in dics_to_use]
    write_path = os.path.join(prefix, 'aa2_bayesian_bf_configs.json')
    with open(write_path, 'w') as fout:
        json.dump(aa2_dics, fout)

    basketball_dic = {
            'dataset_name': 'basketball',
            'n_classes': 2,
            'use_test' : False,
            'n_runs' : 10,
            }
    # we want dataset specific entries to overwrite the general entries
    basketball_dics = [dict(gen_dic, **basketball_dic) for gen_dic in dics_to_use]
    write_path = os.path.join(prefix, 'basketball_bayesian_bf_configs.json')
    with open(write_path, 'w') as fout:
        json.dump(basketball_dics, fout)

    breast_cancer_dic = {
            'dataset_name': 'breast_cancer',
            'n_classes': 2,
            'use_test' : False,
            'n_runs' : 10,
            }
    # we want dataset specific entries to overwrite the general entries
    breast_cancer_dics = [dict(gen_dic, **breast_cancer_dic) for gen_dic in dics_to_use]
    write_path = os.path.join(prefix, 'breast_cancer_bayesian_bf_configs.json')
    with open(write_path, 'w') as fout:
        json.dump(breast_cancer_dics, fout)

    cardio_dic = {
            'dataset_name': 'cardio',
            'n_classes': 2,
            'use_test' : False,
            'n_runs' : 10,
            }
    # we want dataset specific entries to overwrite the general entries
    cardio_dics = [dict(gen_dic, **cardio_dic) for gen_dic in dics_to_use]
    write_path = os.path.join(prefix, 'cardio_bayesian_bf_configs.json')
    with open(write_path, 'w') as fout:
        json.dump(cardio_dics, fout)

    imdb_dic = {
            'dataset_name': 'imdb',
            'n_classes': 2,
            'use_test' : False,
            'n_runs' : 10,
            }
    # we want dataset specific entries to overwrite the general entries
    imdb_dics = [dict(gen_dic, **imdb_dic) for gen_dic in dics_to_use]
    write_path = os.path.join(prefix, 'imdb_bayesian_bf_configs.json')
    with open(write_path, 'w') as fout:
        json.dump(imdb_dics, fout)

    obs_dic = {
            'dataset_name': 'obs',
            'n_classes': 2,
            'use_test' : False,
            'n_runs' : 10,
            }
    # we want dataset specific entries to overwrite the general entries
    obs_dics = [dict(gen_dic, **obs_dic) for gen_dic in dics_to_use]
    write_path = os.path.join(prefix, 'obs_bayesian_bf_configs.json')
    with open(write_path, 'w') as fout:
        json.dump(obs_dics, fout)

    sms_dic = {
            'dataset_name': 'sms',
            'n_classes': 2,
            'use_test' : False,
            'n_runs' : 10,
            }
    # we want dataset specific entries to overwrite the general entries
    sms_dics = [dict(gen_dic, **sms_dic) for gen_dic in dics_to_use]
    write_path = os.path.join(prefix, 'sms_bayesian_bf_configs.json')
    with open(write_path, 'w') as fout:
        json.dump(sms_dics, fout)

    yelp_dic = {
            'dataset_name': 'yelp',
            'n_classes': 2,
            'use_test' : False,
            'n_runs' : 10,
            }
    # we want dataset specific entries to overwrite the general entries
    yelp_dics = [dict(gen_dic, **yelp_dic) for gen_dic in dics_to_use]
    write_path = os.path.join(prefix, 'yelp_bayesian_bf_configs.json')
    with open(write_path, 'w') as fout:
        json.dump(yelp_dics, fout)

    youtube_dic = {
            'dataset_name': 'youtube',
            'n_classes': 2,
            'use_test' : False,
            'n_runs' : 10,
            }
    # we want dataset specific entries to overwrite the general entries
    youtube_dics = [dict(gen_dic, **youtube_dic) for gen_dic in dics_to_use]
    write_path = os.path.join(prefix, 'youtube_bayesian_bf_configs.json')
    with open(write_path, 'w') as fout:
        json.dump(youtube_dics, fout)

    domain_dic = {
            'dataset_name': 'domain',
            'n_classes': 5,
            'use_test' : False,
            'n_runs' : 10,
            }
    # we want dataset specific entries to overwrite the general entries
    domain_dics = [dict(gen_dic, **domain_dic) for gen_dic in dics_to_use]
    write_path = os.path.join(prefix, 'domain_bayesian_bf_configs.json')
    with open(write_path, 'w') as fout:
        json.dump(domain_dics, fout)

    cdr_dic = {
            'dataset_name': 'cdr',
            'n_classes': 2,
            'use_test' : False,
            'n_runs' : 10,
            }
    # we want dataset specific entries to overwrite the general entries
    cdr_dics = [dict(gen_dic, **cdr_dic) for gen_dic in dics_to_use]
    write_path = os.path.join(prefix, 'cdr_bayesian_bf_configs.json')
    with open(write_path, 'w') as fout:
        json.dump(cdr_dics, fout)

    commercial_dic = {
            'dataset_name': 'commercial',
            'n_classes': 2,
            'use_test' : False,
            'n_runs' : 10,
            }
    # we want dataset specific entries to overwrite the general entries
    commercial_dics = [dict(gen_dic, **commercial_dic) for gen_dic in dics_to_use]
    write_path = os.path.join(prefix, 'commercial_bayesian_bf_configs.json')
    with open(write_path, 'w') as fout:
        json.dump(commercial_dics, fout)

    tennis_dic = {
            'dataset_name': 'tennis',
            'n_classes': 2,
            'use_test' : False,
            'n_runs' : 10,
            }
    # we want dataset specific entries to overwrite the general entries
    tennis_dics = [dict(gen_dic, **tennis_dic) for gen_dic in dics_to_use]
    write_path = os.path.join(prefix, 'tennis_bayesian_bf_configs.json')
    with open(write_path, 'w') as fout:
        json.dump(tennis_dics, fout)

    trec_dic = {
            'dataset_name': 'trec',
            'n_classes': 6,
            'use_test' : False,
            'n_runs' : 10,
            }
    # we want dataset specific entries to overwrite the general entries
    trec_dics = [dict(gen_dic, **trec_dic) for gen_dic in dics_to_use]
    write_path = os.path.join(prefix, 'trec_bayesian_bf_configs.json')
    with open(write_path, 'w') as fout:
        json.dump(trec_dics, fout)

    semeval_dic = {
            'dataset_name': 'semeval',
            'n_classes': 9,
            'use_test' : False,
            'n_runs' : 10,
            }
    # we want dataset specific entries to overwrite the general entries
    semeval_dics = [dict(gen_dic, **semeval_dic) for gen_dic in dics_to_use]
    write_path = os.path.join(prefix, 'semeval_bayesian_bf_configs.json')
    with open(write_path, 'w') as fout:
        json.dump(semeval_dics, fout)

    chemprot_dic = {
            'dataset_name': 'chemprot',
            'n_classes': 10,
            'use_test' : False,
            'n_runs' : 10,
            }
    # we want dataset specific entries to overwrite the general entries
    chemprot_dics = [dict(gen_dic, **chemprot_dic) for gen_dic in dics_to_use]
    write_path = os.path.join(prefix, 'chemprot_bayesian_bf_configs.json')
    with open(write_path, 'w') as fout:
        json.dump(chemprot_dics, fout)

    agnews_dic = {
            'dataset_name': 'agnews',
            'n_classes': 4,
            'use_test' : False,
            'n_runs' : 10,
            }
    # we want dataset specific entries to overwrite the general entries
    agnews_dics = [dict(gen_dic, **agnews_dic) for gen_dic in dics_to_use]
    write_path = os.path.join(prefix, 'agnews_bayesian_bf_configs.json')
    with open(write_path, 'w') as fout:
        json.dump(agnews_dics, fout)

    if False:
        synth_dics_to_use = [semisup_kwargs, unsup_uniform_kwargs]
        ### for synthetic datasets
        synth_dic = {
                'dataset_name': 'synth_10p_1000n_100nval__',
                'n_classes': 2,
                'use_test' : False,
                'n_max_labeled' : [-1],
                }

        synth_prefix = os.path.join(prefix, 'synthetic')
        if os.path.exists(synth_prefix):
            n_synth = 10
            for i in range(n_synth):
                synth_dic_i = deepcopy(synth_dic)
                synth_dic_i['dataset_name'] += str(i)
                synth_dics = [dict(gen_dic, **synth_dic_i) for gen_dic in synth_dics_to_use]

                write_path = os.path.join(synth_prefix, 'synth_10p_1000n_100nval__' +\
                        str(i) + '_bayesian_bf_configs.json')
                with open(write_path, 'w') as fout:
                    json.dump(synth_dics, fout)

    ### for crowdsourced datasets
    if False:
        crowd_dics_to_use = [unsup_uniform_kwargs]
        bird_dic = {
                'dataset_name': 'bird',
                'n_classes': 2,
                'use_test' : False,
                'n_runs' : 1,
                }
        # we want dataset specific entries to overwrite the general entries
        bird_dics = [dict(gen_dic, **bird_dic) for gen_dic in crowd_dics_to_use]
        write_path = os.path.join(prefix, 'bird_bayesian_bf_configs.json')
        with open(write_path, 'w') as fout:
            json.dump(bird_dics, fout)

        rte_dic = {
                'dataset_name': 'rte',
                'n_classes': 2,
                'use_test' : False,
                'n_runs' : 1,
                }
        # we want dataset specific entries to overwrite the general entries
        rte_dics = [dict(gen_dic, **rte_dic) for gen_dic in crowd_dics_to_use]
        write_path = os.path.join(prefix, 'rte_bayesian_bf_configs.json')
        with open(write_path, 'w') as fout:
            json.dump(rte_dics, fout)

        dog_dic = {
                'dataset_name': 'dog',
                'n_classes': 4,
                'use_test' : False,
                'n_runs' : 1,
                }
        # we want dataset specific entries to overwrite the general entries
        dog_dics = [dict(gen_dic, **dog_dic) for gen_dic in crowd_dics_to_use]
        write_path = os.path.join(prefix, 'dog_bayesian_bf_configs.json')
        with open(write_path, 'w') as fout:
            json.dump(dog_dics, fout)

        web_dic = {
                'dataset_name': 'web',
                'n_classes': 5,
                'use_test' : False,
                'n_runs' : 1,
                }
        # we want dataset specific entries to overwrite the general entries
        web_dics = [dict(gen_dic, **web_dic) for gen_dic in crowd_dics_to_use]
        write_path = os.path.join(prefix, 'web_bayesian_bf_configs.json')
        with open(write_path, 'w') as fout:
            json.dump(web_dics, fout)
