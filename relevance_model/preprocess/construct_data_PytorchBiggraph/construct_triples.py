import pandas as pd
import numpy as np
import argparse
import logging
import operator
import os
import sys
from rule import Predicate

operators_dict = {'==': operator.eq, '<>': operator.ne, '>': operator.gt, '<': operator.lt, '>=': operator.ge, '<=': operator.le}
comparable_ops = [">", "<", ">=", "<="]


def convert(tuples):  # tuple -> (tid, attr, value)
    src = []
    rel = []
    dst = []
    for index, row in tuples.iterrows():
        for attr, value in row.items():
            if pd.isnull(value) or str(value) == "nan" or value is None:
                continue
            src.append("t"+str(index))
            rel.append(attr)
            dst.append(value)
    triples_tuples = {"src": src, "rel": rel, "dst": dst}
    triples_tuples = pd.DataFrame(triples_tuples)
    return triples_tuples


def pick_constants(tuples, all_attrs, constant_ratio_low, constant_ratio_high):
    total_size = tuples.shape[0]
    size_lower_bound = total_size * 1.0 * constant_ratio_low
    size_upper_bound = total_size * 1.0 * constant_ratio_high
    constants_filtered = {}
    for attr in all_attrs:
        value2times = tuples[attr].value_counts()
        values = value2times[(value2times >= size_lower_bound) & (value2times <= size_upper_bound)].index.values
        constants_filtered[attr] = list(values)
    return constants_filtered


def enumerate_predicates(predicate_operators, tid_num, all_attrs, comparable_attrs, constants_filtered, data_name):
    src_tvid, src_attr_value = [], []
    rel_tvid, rel_attr_value = [], []
    dst_tvid, dst_attr_value = [], []
    all_predicates = []
    for t_i in range(tid_num):
        for attr in all_attrs:
            for op in predicate_operators:
                if attr not in comparable_attrs and op in comparable_ops:
                    continue
                # (1) enumerate constant predicate
                for constant in constants_filtered[attr]:
                    pre = Predicate(data_name)
                    pre.assign_info(t_i, None, attr, "", constant, op, "constant")
                    all_predicates.append(pre)
                    # record triples for tuple variable id: (tuple_variable_id, predicate, tuple_variable_id)
                    src_tvid.append("tv" + str(t_i))
                    rel_tvid.append(pre.print_predicate_new())
                    dst_tvid.append("tv" + str(t_i))
                    # record triples for attr and value: (attr, predicate, value)
                    src_attr_value.append(attr)
                    rel_attr_value.append(pre.print_predicate_new())
                    dst_attr_value.append(constant)
                # (2) enumerate non-constant predicate
                for t_j in range(t_i+1, tid_num):
                    pre = Predicate(data_name)
                    pre.assign_info(t_i, t_j, attr, attr, "", op, "non-constant")
                    all_predicates.append(pre)
                    # record triples for tuple variable id: (tuple_variable_id, predicate, tuple_variable_id)
                    src_tvid.append("tv" + str(t_i))
                    rel_tvid.append(pre.print_predicate_new())
                    dst_tvid.append("tv" + str(t_j))
                    # record triples for attr and value: (attr1, predicate, attr2)
                    src_attr_value.append(attr)
                    rel_attr_value.append(pre.print_predicate_new())
                    dst_attr_value.append(attr)

    triples_tvid = {"src": src_tvid, "rel": rel_tvid, "dst": dst_tvid}
    triples_tvid = pd.DataFrame(triples_tvid)

    triples_attr_value = {"src": src_attr_value, "rel": rel_attr_value, "dst": dst_attr_value}
    triples_attr_value = pd.DataFrame(triples_attr_value)

    return all_predicates, triples_tvid, triples_attr_value


def print_all_predicates(all_predicates):
    print("----------------- all predicates -----------------")
    for pre in all_predicates:
        print(pre.print_predicate_new())
    print("--------------------------------------------------")


def compute_satisfied_tuple_ids(tuples, all_predicates):
    src_tid = []
    rel_tid = []
    dst_tid = []
    for pre in all_predicates:
        attr1 = pre.get_attr1()
        attr2 = pre.get_attr2()
        constant = pre.get_constant()
        op = pre.get_operator()
        if pre.get_type() == "constant":
            satisfied_indices = tuples.loc[operators_dict[op](tuples[attr1], constant)].index.values
            src_tid = src_tid + ["t" + str(idx) for idx in satisfied_indices]
            rel_tid = rel_tid + [pre.print_predicate_new()] * satisfied_indices.shape[0]
            dst_tid = dst_tid + ["t" + str(idx) for idx in satisfied_indices]
        elif pre.get_type() == "non-constant":
            value2tid = {}
            for value, df in tuples.groupby(attr1):
                if op == "=" and df.shape[0] == 1:
                    continue
                value2tid[value] = list(df.index)
            all_values = list(value2tid.keys())
            if op == "=":
                for value in all_values:
                    indices = value2tid[value]
                    for i in range(len(indices)-1):
                        for j in range(i+1, len(indices)):
                            src_tid.append("t" + str(indices[i]))
                            rel_tid.append(pre.print_predicate_new())
                            dst_tid.append("t" + str(indices[j]))
            else:  # <>, >, >=, <, <=
                for i in range(len(all_values)-1):
                    value1 = all_values[i]
                    indices1 = value2tid[value1]
                    for j in range(i+1, len(all_values)):
                        value2 = all_values[j]
                        indices2 = value2tid[value2]
                        if not operators_dict[op](value1, value2):
                            continue
                        for idx1 in indices1:
                            for idx2 in indices2:
                                src_tid.append("t" + str(idx1))
                                rel_tid.append(pre.print_predicate_new())
                                dst_tid.append("t" + str(idx2))

    triples_tid = {"src": src_tid, "rel": rel_tid, "dst": dst_tid}
    triples_tid = pd.DataFrame(triples_tid)

    return triples_tid

def obtain_operator(predicate):
    operator = ''
    if (predicate.find('<>') != -1):
        operator = '<>'
    elif (predicate.find('>=') != -1):
        operator = '>='
    elif (predicate.find('<=') != -1):
        operator = '<='
    elif (predicate.find('==') != -1):
        operator = '=='
    elif (predicate.find('=') != -1):
        operator = '='
    elif (predicate.find('>') != -1):
        operator = '>'
    elif (predicate.find('<') != -1):
        operator = '<'
    return operator


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Given a relational data(.csv), convert it to the triples.")
    parser.add_argument("-data_dir", "--data_dir", type=str, default="../../datasets/airports/")
    parser.add_argument("-data_name", "--data_name", type=str, default="airports")
    parser.add_argument("-tuple_variable_num", "--tuple_variable_num", type=int, default=2)
    parser.add_argument("-predicate_operators", "--predicate_operators", type=str, default="==")  # "==,<>,>,>=,<,<="
    parser.add_argument("-comparable_attrs", "--comparable_attrs", type=str, default="")  # "ZIP_Code"
    parser.add_argument("-constant_ratio_low", "--constant_ratio_low", type=float, default=0.01)  # 0.02 for ncvoter
    parser.add_argument("-constant_ratio_high", "--constant_ratio_high", type=float, default=1.0)
    parser.add_argument("-seed", "--seed", type=int, default=42)

    args = parser.parse_args()
    arg_dict = args.__dict__
    for k, v in sorted(arg_dict.items()):
        logging.info('[Argument] %s=%r' % (k, v))

    data_dir = arg_dict["data_dir"]
    data_path = data_dir + arg_dict["data_name"] + "_sample.csv"
    data = pd.read_csv(data_path)
    all_attrs = data.columns.tolist()
    print("data shape: ", data.shape)
    print("all_attrs: ", all_attrs)

    revised_attrs = None
    if "airports" in arg_dict["data_name"]:
        revised_attrs = ["id", "elevation_ft"]
    elif "hospital" in arg_dict["data_name"]:
        revised_attrs = ["ZIP_Code", "Phone_Number"]
    elif "ncvoter" in arg_dict["data_name"]:
        revised_attrs = ["id", "county_id"]
    elif "tax" in arg_dict["data_name"]:
        revised_attrs = ["areacode", "phone", "zip", "salary", "singleexemp", "marriedexemp", "childexemp"]
    elif "AMiner_Author2Paper" in arg_dict["data_name"]:
        revised_attrs = ["author2paper_id", "author_id", "paper_id", "author_position"]
    elif "AMiner_Author" in arg_dict["data_name"]:
        revised_attrs = ["author_id", "published_papers", "citations", "h_index"]
    elif "AMiner_Paper" in arg_dict["data_name"]:
        revised_attrs = ["paper_id", "year"]
    elif "adults_categorical" in arg_dict["data_name"]:
        revised_attrs = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
    elif "adults" in arg_dict["data_name"]:
        revised_attrs = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
        
    if revised_attrs is not None:
        for attr in revised_attrs:
            data[attr] = data[attr].astype(str).str.replace('\.0$', '').replace("nan", np.nan)

    # convert relational data to triples, and save it.
    triples_tuples_path = data_dir + "triples_tuples.csv"
    if not os.path.exists(triples_tuples_path):
        triples_tuples = convert(data)
        triples_tuples = triples_tuples.sample(frac=1, random_state=arg_dict["seed"]).reset_index(drop=True)
        triples_tuples.to_csv(triples_tuples_path, sep="\t", index=False, header=False)
    else:
        triples_tuples = pd.read_csv(triples_tuples_path, sep="\t", names=["src", "rel", "dst"])

    # count and pick constants for constant predicate enumeration
    constants_filtered = pick_constants(data, all_attrs, arg_dict["constant_ratio_low"], arg_dict["constant_ratio_high"])

    # enumerate all predicates, and save triples related to predicates.
    predicate_operators = arg_dict["predicate_operators"].split(",")
    comparable_attrs = arg_dict["comparable_attrs"].split("||")
    all_predicates_path = data_dir + "all_predicates.txt"
    triples_tvid_path = data_dir + "triples_tvid.csv"
    triples_attr_value_path = data_dir + "triples_attr_value.csv"
    if not os.path.exists(triples_tvid_path) or not os.path.exists(triples_attr_value_path):
        all_predicates, triples_tvid, triples_attr_value = enumerate_predicates(predicate_operators,
                                                                                arg_dict["tuple_variable_num"],
                                                                                all_attrs, comparable_attrs,
                                                                                constants_filtered,
                                                                                arg_dict["data_name"])
        triples_tvid = triples_tvid.sample(frac=1, random_state=arg_dict["seed"]).reset_index(drop=True)
        triples_attr_value = triples_attr_value.sample(frac=1, random_state=arg_dict["seed"]).reset_index(drop=True)
        print_all_predicates(all_predicates)
        triples_tvid.to_csv(triples_tvid_path, sep="\t", index=False, header=False)
        triples_attr_value.to_csv(triples_attr_value_path, sep="\t", index=False, header=False)
        f = open(all_predicates_path, "w")
        for pre in all_predicates:
            f.writelines(pre.print_predicate_new() + "\n")
        f.close()
    else:
        triples_tvid = pd.read_csv(triples_tvid_path, sep="\t", names=["src", "rel", "dst"])
        triples_attr_value = pd.read_csv(triples_attr_value_path, sep="\t", names=["src", "rel", "dst"])
        all_predicates = []
        f = open(all_predicates_path, "r")
        lines = f.readlines()
        for line in lines:
            operator = obtain_operator(line)
            pre = Predicate(arg_dict["data_name"])
            pre.transform(line.strip(), operator)
            all_predicates.append(pre)
        f.close()

    # get satisfied tuple ids for predicates, and save the corresponding triplets.
    triples_tid_path = data_dir + "triples_tid.csv"
    if not os.path.exists(triples_tid_path):
        triples_tid = compute_satisfied_tuple_ids(data, all_predicates)
        triples_tid = triples_tid.sample(frac=1, random_state=arg_dict["seed"]).reset_index(drop=True)
        triples_tid.to_csv(triples_tid_path, sep="\t", index=False, header=False)
    else:
        triples_tid = pd.read_csv(triples_tid_path, sep="\t", names=["src", "rel", "dst"])
