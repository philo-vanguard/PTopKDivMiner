import copy
import pandas as pd
import argparse
import logging
from rule import REELogic
import random
import os
import numpy as np


def load_rules(path):
    f = open(path, "r")
    lines = f.readlines()
    f.close()

    rees = []
    for line in lines:
        if "->" not in line:
            continue
        ree = REELogic()
        ree.load_X_and_e(line)
        rees.append(ree)
        print(ree.print_rule())

    print("finish loading, ", len(rees), " rules")
    return rees


def count_violations(data, rees):
    isViolateInfo = []
    ree_count = 0
    for ree in rees:
        if ree_count % 10 == 0:
            print("process ", ree_count, " rees")
        isViolate = pd.Series([False] * data.shape[0])
        isViolate.index = data.index
        rhs = ree.get_RHS()
        check_attr = rhs.get_attr1()
        value2tid_tuples, value2tid_tuples_2 = {}, {}
        constants_in_X = [[], []]  # for t0, t1
        # get key 
        key_attributes_non_constant = []
        for predicate in ree.get_currents():
            if not predicate.is_constant():
                key_attributes_non_constant.append(predicate.get_attr1())
            else:
                pre_tid = predicate.get_index1()
                constants_in_X[pre_tid].append((predicate.get_attr1(), predicate.get_constant()))  # (A, a)
        # get value of tuples
        # (1) first filter tuples that not satisfy the constant predicates
        reserved_indices = isViolate.loc[isViolate == False].index.values
        for attr, v in constants_in_X[0]:  # constant predicates
            reserved_indices = data.loc[data[attr].astype(str) == v].index.values
            if reserved_indices.shape[0] == 0:
                break
        if reserved_indices.shape[0] == 0:  # there's no tuples satisfy the constant predicates in X of current ree; we should go for the next ree
            isViolateInfo.append(isViolate.loc[isViolate == True].index.values.shape[0])
            continue
        # (2) then construct dict by non-constant predicates
        for value, df in data.loc[reserved_indices].groupby(key_attributes_non_constant):
            if df.shape[0] == 1:
                continue
            value2tid_tuples[value] = df
        if len(value2tid_tuples) == 0:
            isViolateInfo.append(isViolate.loc[isViolate == True].index.values.shape[0])
            continue
        # get value of tuples again
        # (1) first filter tuples that not satisfy the constant predicates
        reserved_indices_2 = isViolate.loc[isViolate == False].index.values
        for attr, v in constants_in_X[1]:  # constant predicates
            reserved_indices_2 = data.loc[data[attr].astype(str) == v].index.values
            if reserved_indices_2.shape[0] == 0:
                break
        if reserved_indices_2.shape[0] == 0:  # there's no tuples satisfy the constant predicates in X of current ree; we should go for the next ree
            isViolateInfo.append(isViolate.loc[isViolate == True].index.values.shape[0])
            continue
        # (2) then construct dict by non-constant predicates
        for value, df in data.loc[reserved_indices_2].groupby(key_attributes_non_constant):
            if df.shape[0] == 1:
                continue
            value2tid_tuples_2[value] = df
        if len(value2tid_tuples_2) == 0:
            isViolateInfo.append(isViolate.loc[isViolate == True].index.values.shape[0])
            continue
        # apply constant Y
        if rhs.get_type() == "constant" and pd.notnull(rhs.get_constant()):
            for key in value2tid_tuples.keys():
                if key not in value2tid_tuples_2.keys():
                    continue
                indices = value2tid_tuples[key].index.values
                indices_2 = value2tid_tuples_2[key].index.values

                if rhs.get_index1() == 0:
                    check_tuples = data.loc[indices]
                else:
                    check_tuples = data.loc[indices_2]
                violate_t_in_data = check_tuples.loc[check_tuples[check_attr].astype(str) != rhs.get_constant()].index.values
                isViolate.loc[violate_t_in_data] = True
        # apply non-constant Y
        else:
            for key in value2tid_tuples.keys():
                if key not in value2tid_tuples_2.keys():
                    continue
                indices = value2tid_tuples[key].index.values
                indices_2 = value2tid_tuples_2[key].index.values
                if indices.shape[0] <= indices_2.shape[0]:
                    for index in indices:
                        value = data.loc[index, check_attr]
                        if pd.notnull(value) and value is not None and str(value) != "":
                            irredundant_index_2 = copy.deepcopy(list(indices_2))
                            if index in irredundant_index_2:
                                irredundant_index_2.remove(index)
                            value_2 = data.loc[irredundant_index_2, check_attr]
                            vio_index_in_data = value_2.loc[value_2 != value].index.values
                            isViolate.loc[vio_index_in_data] = True
                else:
                    for index_2 in indices_2:
                        value_2 = data.loc[index_2, check_attr]
                        if pd.notnull(value_2) and value_2 is not None and str(value_2) != "":
                            irredundant_index = copy.deepcopy(list(indices))
                            if index_2 in irredundant_index:
                                irredundant_index.remove(index_2)
                            value = data.loc[irredundant_index, check_attr]
                            vio_index_in_data = value.loc[value != value_2].index.values
                            isViolate.loc[vio_index_in_data] = True

        violate_indices = isViolate.loc[isViolate == True].index.values
        isViolateInfo.append(violate_indices.shape[0])

        ree_count = ree_count + 1

    return isViolateInfo


def construct_rule_pairs(all_rules, num_pairs, train_ratio, isViolateInfo):
    # randomly choose some rule pairs, and label them according to the satisfaction and violations
    random.seed(1234567)
    size = len(all_rules)
    rule_pairs = set()
    while len(rule_pairs) < num_pairs:
        id1 = random.randint(0, size-1)
        id2 = random.randint(0, size-1)
        while id1 == id2:
            id2 = random.randint(0, size-1)
        if isViolateInfo[id1] > isViolateInfo[id2]:
            if id1 < id2:
                rule_pairs.add((id1, id2, 0))
            else:
                rule_pairs.add((id2, id1, 1))
        elif isViolateInfo[id1] < isViolateInfo[id2]:
            if id1 < id2:
                rule_pairs.add((id1, id2, 1))
            else:
                rule_pairs.add((id2, id1, 0))
        else:
            continue

    rule_pairs = pd.DataFrame(rule_pairs)
    rule_pairs.columns = ["left_id", "right_id", "label"]
    rule_pairs.index = range(num_pairs)

    num_train = int(num_pairs * train_ratio)
    rule_pairs_train = rule_pairs.loc[:num_train, :]
    rule_pairs_test = rule_pairs.loc[num_train:, :]

    return rule_pairs_train, rule_pairs_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="label the given rule pairs")
    parser.add_argument('-data_name', '--data_name', type=str, default="airports")
    parser.add_argument('-rule_path', '--rule_path', type=str, default="../datasets/airports/rules.txt")
    parser.add_argument('-clean_data', '--clean_data', type=str, default="../datasets/airports/airports_sample_clean.csv")
    parser.add_argument('-num_pairs', '--num_pairs', type=int, default=1000)
    parser.add_argument('-train_ratio', '--train_ratio', type=float, default=0.8)
    parser.add_argument('-train_test_data_dir', '--train_test_data_dir', type=str, default="../datasets/airports/train/")

    args = parser.parse_args()
    arg_dict = args.__dict__
    for k, v in sorted(arg_dict.items()):
        logging.info('[Argument] %s=%r' % (k, v))
        print("k:", k, ", v:", v)

    # load data
    clean_data = pd.read_csv(arg_dict["clean_data"])

    # transform float to int
    revised_attrs = None
    if arg_dict["data_name"] == "airports":
        revised_attrs = ["id", "elevation_ft"]
    elif arg_dict["data_name"] == "ncvoter":
        revised_attrs = ["county_id", "id"]
    elif arg_dict["data_name"] == "hospital":
        revised_attrs = ["ZIP_Code", "Phone_Number"]
    elif "tax" in arg_dict["data_name"]:
        revised_attrs = ["areacode", "phone", "zip", "salary", "singleexemp", "marriedexemp", "childexemp"]
    elif "aminer" in arg_dict["data_name"]:
        if "AMiner_Author2Paper" in arg_dict["clean_data"]:
            revised_attrs = ["author2paper_id", "author_id", "paper_id", "author_position"]
        elif "AMiner_Author" in arg_dict["clean_data"]:
            revised_attrs = ["author_id", "published_papers", "citations", "h_index"]
        elif "AMiner_Paper" in arg_dict["clean_data"]:
            revised_attrs = ["paper_id", "year"]
    elif "adults_categorical" in arg_dict["data_name"]:
        revised_attrs = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
    elif "adults" in arg_dict["data_name"]:
        revised_attrs = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
    
    if revised_attrs is not None:
        for attr in revised_attrs:
            clean_data[attr] = clean_data[attr].astype(str).str.replace('\.0$', '').replace("nan", np.nan)

    # load rules
    rees = load_rules(arg_dict["rule_path"])

    # apply rules on the clean data, to count the violations
    isViolateInfo = count_violations(clean_data, rees)
    print("rule size:", len(rees), ", len isViolateInfo:", len(isViolateInfo))
    print("the number of violations for each rule:\n", isViolateInfo)

    # generate datasets
    rule_pairs_train, rule_pairs_test = construct_rule_pairs(rees, arg_dict["num_pairs"], arg_dict["train_ratio"], isViolateInfo)

    # write into file
    if not os.path.exists(arg_dict["train_test_data_dir"]):
        os.mkdir(arg_dict["train_test_data_dir"])
    rule_pairs_train.to_csv(arg_dict["train_test_data_dir"] + "train.csv", sep=",", index=False)
    rule_pairs_test.to_csv(arg_dict["train_test_data_dir"] + "test.csv", sep=",", index=False)
