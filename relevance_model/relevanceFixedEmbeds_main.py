import argparse
import logging

import numpy as np
from sklearn.model_selection import train_test_split
from relevanceFixedEmbeds import *
from utils import *
import pandas as pd
import pickle as plk
import os
from transformers import BertTokenizer, BertModel
import torch


def main():
    parser = argparse.ArgumentParser(description="Learn the relevance model")
    parser.add_argument('-lr', '--lr', type=float, default=0.001)
    parser.add_argument('-hidden_size', '--hidden_size', type=int, default=100)
    parser.add_argument('-rees_embed_dim', '--rees_embed_dim', type=int, default=100)
    parser.add_argument('-epochs', '--epochs', type=int, default=400)
    parser.add_argument('-batch_size', '--batch_size', type=int, default=128)
    parser.add_argument('-pretrained_matrix_file', '--pretrained_matrix_file', type=str, default='./datasets/airports/predicateEmbedds.csv')
    parser.add_argument('-rules_file', '--rules_file', type=str, default='./datasets/airports/rules.txt')
    parser.add_argument('-train_file', '--train_file', type=str, default='./datasets/airports/train/train.csv')
    parser.add_argument('-test_file', '--test_file', type=str, default='./datasets/airports/train/test.csv')
    parser.add_argument('-withBertInitialization', '--withBertInitialization', type=int, default=0)  # 0: initialize by Pytorch-Biggraph; 1: initialize by Bert
    parser.add_argument('-output_model_dir', '--output_model_dir', type=str, default='./datasets/airports/train/model/')  # 0: model; 1: model_initializedByBert
    parser.add_argument('-all_predicates_file', '--all_predicates_file', type=str, default='./datasets/airports/all_predicates.txt')
    parser.add_argument('-modelOpt', '--modelOpt', type=str, default="distilbert-base-uncased")
    parser.add_argument('-times', '--times', type=str, default="1")


    args = parser.parse_args()
    arg_dict = args.__dict__
    for k, v in sorted(arg_dict.items()):
        logging.info('[Argument] %s=%r' % (k, v))
        print("k:", k, ", v:", v)

    if not os.path.exists(arg_dict['output_model_dir']):
        os.mkdir(arg_dict['output_model_dir'])

    # load data
    train_data = pd.read_csv(arg_dict['train_file'])
    test_data = pd.read_csv(arg_dict['test_file'])
    rule_data_header = ["rule","support","confidence","relevance_score","diversity_score","score"]
    rees_data = pd.read_csv(arg_dict['rules_file'], names=rule_data_header)["rule"].values

    # split train and valid
    train_data, valid_data = train_test_split(train_data, train_size=0.9, random_state=42)
    train_pair_ids, train_labels = train_data[['left_id', 'right_id']].values, train_data['label'].values
    valid_pair_ids, valid_labels = valid_data[['left_id', 'right_id']].values, valid_data['label'].values
    test_pair_ids, test_labels = test_data[['left_id', 'right_id']].values, test_data['label'].values
    train_labels = convert_to_onehot(train_labels)
    valid_labels = convert_to_onehot(valid_labels)
    test_labels = convert_to_onehot(test_labels)

    # load pretrained matrix, i.e., the initial embeddings of predicates
    pretrained_matrix = pd.read_csv(arg_dict['pretrained_matrix_file'])
    predicate_embedding_size = pretrained_matrix.loc[0, "predicate_embedding_size"]
    print("pretrained_matrix shape:", pretrained_matrix.shape)

    ####  write all predicates into file
    all_predicates_str = pretrained_matrix["predicate"].values.tolist()
    all_predicates_str.insert(0, "PAD")
    f = open(arg_dict['all_predicates_file'], "w")
    for predicate_str in all_predicates_str:
        f.write(predicate_str)
        f.write("\n")
    f.close()

    all_predicate_size = len(all_predicates_str)  # includes 'PAD'
    print("all_predicate_size len: ", all_predicate_size)

    #### transform pretrained matrix
    if arg_dict["withBertInitialization"] == 0:
        predicate_initial_embedds = pretrained_matrix.loc[:, "0":].values
        pad_embedding = np.zeros((1, predicate_embedding_size))
        predicate_initial_embedds = np.concatenate((pad_embedding, predicate_initial_embedds), axis=0)
        print(predicate_initial_embedds.shape[0])

    else:
        # predicate embeddings initialized by Bert
        predicate_embedding_size = 768
        tokenizer = BertTokenizer.from_pretrained(arg_dict["modelOpt"])
        model = BertModel.from_pretrained(arg_dict["modelOpt"])

        encoded_input = tokenizer(all_predicates_str, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            output = model(**encoded_input)
        predicate_initial_embedds = output.last_hidden_state.mean(dim=1)
        print(predicate_initial_embedds.shape[0])
    
    print("predicate_embedding_size:", predicate_embedding_size)

    rees_lhs, rees_rhs = processAllRulesByPredicates(rees_data, all_predicates_str)
    print('The first 10 data')
    print(rees_lhs[:5])
    print(rees_rhs[:5])

    # model
    model = RelevanceEmbeds(all_predicate_size, predicate_embedding_size, arg_dict['hidden_size'],
                          arg_dict['rees_embed_dim'], MAX_LHS_PREDICATES,
                          MAX_RHS_PREDICATES, arg_dict['lr'], arg_dict['epochs'], arg_dict['batch_size'],
                          predicate_initial_embedds)

    # train
    model.train(rees_lhs, rees_rhs, train_pair_ids, train_labels, valid_pair_ids, valid_labels)
    # evaluate
    test_log = model.evaluate(rees_lhs, rees_rhs, test_pair_ids, test_labels)
    f = open(arg_dict['output_model_dir'] + "acc-" + arg_dict["times"] + ".txt", "w")
    f.write(test_log)
    f.close()
    # save the rule interestingness model in TensorFlow type
    model.saveModel(arg_dict['output_model_dir'] + "model")
    # save the rule interestingness model in TXT type
    model.saveModelToText(arg_dict['output_model_dir'] + "model.txt")

    # predict the relevance score of each rule
    relevance_scores = model.predicate_relevance_scores(rees_lhs, rees_rhs)
    relevance_scores = [relevance_scores[i][0][0] for i in range(len(relevance_scores))]
    print("max rel score:", max(relevance_scores))
    rees_results = pd.concat([pd.Series(rees_data), pd.Series(relevance_scores)], axis=1)
    rees_results.to_csv(arg_dict['output_model_dir'] + "relevance_scores.csv", index=False, header=False)
    print("rees_results: ", rees_results)


if __name__ == '__main__':
    main()

