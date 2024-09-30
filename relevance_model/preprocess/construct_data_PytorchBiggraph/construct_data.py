import os

import pandas as pd
import argparse
import logging


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Given all triple files of a relational data(.csv), merge triples and split into train, valid, test data.")
    parser.add_argument("-data_dir", "--data_dir", type=str, default="../../datasets/airports/")
    parser.add_argument("-train_ratio", "--train_ratio", type=float, default=0.8)
    parser.add_argument("-valid_ratio", "--valid_ratio", type=float, default=0.1)
    parser.add_argument("-test_ratio", "--test_ratio", type=float, default=0.1)

    args = parser.parse_args()
    arg_dict = args.__dict__
    for k, v in sorted(arg_dict.items()):
        logging.info('[Argument] %s=%r' % (k, v))

    data_dir = arg_dict["data_dir"]
    triples_tuples = pd.read_csv(data_dir + "triples_tuples.csv", sep="\t", names=["src", "rel", "dst"])
    # triples_tvid = pd.read_csv(data_dir + "triples_tvid.csv", sep="\t", names=["src", "rel", "dst"])
    triples_attr_value = pd.read_csv(data_dir + "triples_attr_value.csv", sep="\t", names=["src", "rel", "dst"])
    triples_tid = pd.read_csv(data_dir + "triples_tid.csv", sep="\t", names=["src", "rel", "dst"])

    # merge all triples, and split into train, valid, test data
    train_data = []
    valid_data = []
    test_data = []
    # (1) triples_tuples
    num_triples = triples_tuples.shape[0]
    num_train = int(num_triples * arg_dict["train_ratio"])
    num_valid = int(num_triples * arg_dict["valid_ratio"])
    train_data.append(triples_tuples.iloc[:num_train])
    valid_data.append(triples_tuples.iloc[num_train:(num_train+num_valid)])
    test_data.append(triples_tuples.iloc[(num_train+num_valid):])

    # (2) triples_tvid
    # num_triples = triples_tvid.shape[0]
    # num_train = int(num_triples * arg_dict["train_ratio"])
    # num_valid = int(num_triples * arg_dict["valid_ratio"])
    # train_data.append(triples_tvid.iloc[:num_train])
    # valid_data.append(triples_tvid.iloc[num_train:(num_train + num_valid)])
    # test_data.append(triples_tvid.iloc[(num_train + num_valid):])

    # (3) triples_attr_value
    num_triples = triples_attr_value.shape[0]
    num_train = int(num_triples * arg_dict["train_ratio"])
    num_valid = int(num_triples * arg_dict["valid_ratio"])
    train_data.append(triples_attr_value.iloc[:num_train])
    valid_data.append(triples_attr_value.iloc[num_train:(num_train + num_valid)])
    test_data.append(triples_attr_value.iloc[(num_train + num_valid):])

    # (4) triples_tid
    num_triples = triples_tid.shape[0]
    num_train = int(num_triples * arg_dict["train_ratio"])
    num_valid = int(num_triples * arg_dict["valid_ratio"])
    train_data.append(triples_tid.iloc[:num_train])
    valid_data.append(triples_tid.iloc[num_train:(num_train+num_valid)])
    test_data.append(triples_tid.iloc[(num_train+num_valid):])

    # (5) merge
    train_df = train_data[0]
    valid_df = valid_data[0]
    test_df = test_data[0]
    for i in range(1, len(train_data)):
        train_df = pd.concat([train_df, train_data[i]])
        valid_df = pd.concat([valid_df, valid_data[i]])
        test_df = pd.concat([test_df, test_data[i]])

    # replace " " with "_"
    for attr in ["src", "rel", "dst"]:
        train_df[attr] = train_df[attr].apply(lambda x: str(x).replace(" ", "##").replace("\n", "##"))
        valid_df[attr] = valid_df[attr].apply(lambda x: str(x).replace(" ", "##").replace("\n", "##"))
        test_df[attr] = test_df[attr].apply(lambda x: str(x).replace(" ", "##").replace("\n", "##"))

    # write into file
    if not os.path.exists(data_dir + "/train_pytorchBiggraph/"):
        os.mkdir(data_dir + "/train_pytorchBiggraph/")
    train_df.to_csv(data_dir + "/train_pytorchBiggraph/train.csv", sep="\t", index=False, header=False)
    valid_df.to_csv(data_dir + "/train_pytorchBiggraph/valid.csv", sep="\t", index=False, header=False)
    test_df.to_csv(data_dir + "/train_pytorchBiggraph/test.csv", sep="\t", index=False, header=False)
