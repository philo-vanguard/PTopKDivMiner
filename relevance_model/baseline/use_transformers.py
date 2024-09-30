import os
import pandas as pd
import numpy as np
import copy
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn import metrics
import argparse

os.environ["WANDB_DISABLED"] = "true"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def str_to_bool(str):
    return True if str.lower() == 'true' else False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_name', '--data_name', type=str, default="airports")
    parser.add_argument('-rule_path', '--rule_path', type=str, default="../datasets/airports/rules.txt")
    parser.add_argument('-train_data', '--train_data', type=str, default="../datasets/airports/train/train.csv")
    parser.add_argument('-test_data', '--test_data', type=str, default="../datasets/airports/train/test.csv")
    parser.add_argument('-mlmOption', '--mlmOption', type=str_to_bool, default=False)
    parser.add_argument('-cuda', '--cuda', type=str, default="1")
    parser.add_argument('-modelOpt', '--modelOpt', type=str, default="distilbert-base-uncased")
    parser.add_argument('-batch_size', '--batch_size', type=int, default=100)
    parser.add_argument('-train_epochs', '--train_epochs', type=int, default=50)
    parser.add_argument('-learning_rate', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-output_dir', '--output_dir', type=str, default="../datasets/airports/train/models_baseline_bert/")
    parser.add_argument('-times', '--times', type=str, default="1")
    args = parser.parse_args()
    arg_dict = args.__dict__

    data_name = arg_dict["data_name"]
    os.environ['CUDA_VISIBLE_DEVICES'] = arg_dict["cuda"]

    saved_model_path = arg_dict["output_dir"]

    if not os.path.exists(saved_model_path):
        os.mkdir(saved_model_path)

    rule_data_header = ["rule", "support", "confidence", "relevance_score", "diversity_score", "score"]
    all_rules_set = pd.read_csv(arg_dict["rule_path"], names=rule_data_header)
    all_rules_set = all_rules_set.values

    if not os.path.exists(arg_dict["output_dir"] + 'rule_pairs_train.csv'):
        rules_pair_train = pd.read_csv(arg_dict["train_data"])
        rules_pair_test = pd.read_csv(arg_dict["test_data"])

        # split rule pairs into training, valid, and testing data
        rules_pair_train = rules_pair_train.sample(frac=1.0, random_state=1234567).reset_index(drop=True)
        total_size = rules_pair_train.shape[0]
        train_size = int(total_size * 0.9)
        train_data = rules_pair_train.loc[:train_size-1, :]
        valid_data = rules_pair_train.loc[train_size:, :]
        test_data = rules_pair_test
        train_data.to_csv(arg_dict["output_dir"] + "train.csv", index=False)
        valid_data.to_csv(arg_dict["output_dir"] + "valid.csv", index=False)
        test_data.to_csv(arg_dict["output_dir"] + "test.csv", index=False)

        for option in ('train', 'valid', 'test'):
            rules_pair_ids_set = pd.read_csv(arg_dict["output_dir"] + option + ".csv")
            # prepare the rule pairs
            rules_pairs_set = []
            for left_id, right_id, label in rules_pair_ids_set.values:
                rule_left, rule_right = all_rules_set[int(left_id)][0], all_rules_set[int(right_id)][0]
                rules_pairs_set.append([rule_left, rule_right, int(label)])
            rules_pairs_set = np.array(rules_pairs_set)
            # save csv
            rules_pairs_df = pd.DataFrame(rules_pairs_set, columns=['rule_left', 'rule_right', 'label'])
            rules_pairs_df.to_csv(arg_dict["output_dir"] + 'rule_pairs_' + option + '.csv', index=False)

    _datasets = []
    for option in ('train', 'valid', 'test'):
        tmp_dataset = load_dataset('csv', data_files=arg_dict["output_dir"] + 'rule_pairs_' + option + '.csv')
        _datasets.append(tmp_dataset)

    train_dataset, valid_dataset, test_dataset = _datasets

    dataset = train_dataset
    dataset['validation'] = valid_dataset['train']
    dataset['test'] = test_dataset['train']

    tokenizer = AutoTokenizer.from_pretrained(arg_dict["modelOpt"], use_fast=True)

    rule1_key, rule2_key = 'rule_left', 'rule_right'

    def preprocess_function(examples):
        return tokenizer(examples[rule1_key], examples[rule2_key], truncation=True)

    # encoded_dataset = dataset.map(preprocess_function)
    encoded_dataset = dataset.map(preprocess_function, batched=True)

    num_labels = 2
    model = AutoModelForSequenceClassification.from_pretrained(arg_dict["modelOpt"], num_labels=num_labels)

    metric = 'accuracy'
    model_name = arg_dict["modelOpt"].split("/")[-1]

    model_args = TrainingArguments(output_dir=saved_model_path,
                            evaluation_strategy='epoch',
                            save_strategy='epoch',
                            learning_rate=2e-5,
                            per_device_train_batch_size=arg_dict["batch_size"],
                            per_device_eval_batch_size=arg_dict["batch_size"],
                            num_train_epochs=arg_dict["train_epochs"],
                            weight_decay=0.01,
                            load_best_model_at_end=True,
                            metric_for_best_model='accuracy',
                            push_to_hub=False)


    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        # print(predictions, labels)
        precision = metrics.precision_score(labels, predictions)
        recall = metrics.recall_score(labels, predictions)
        f1 = metrics.f1_score(labels, predictions)
        acc = metrics.accuracy_score(labels, predictions)
        return {'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': acc}


    trainer = Trainer(model, model_args, train_dataset=encoded_dataset['train'], eval_dataset=encoded_dataset['validation'], tokenizer=tokenizer, compute_metrics=compute_metrics)

    trainer.train()

    trainer.evaluate()

    ll = trainer.evaluate(encoded_dataset['test'])
    print(ll)

    f = open(arg_dict["output_dir"] + "results-" + arg_dict["times"] + ".txt", "w")
    f.write(str(ll))
    f.close()

    trainer.save_model()

