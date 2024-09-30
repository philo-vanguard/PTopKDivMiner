#!/bin/bash

dir=$1
cuda=$2
data_dir=${dir}"relevance_model/datasets/"
code_dir=${dir}"relevance_model/baseline/"

for data_name in "airports" "hospital" "inspection" "ncvoter" "dblp" "adults"
do
    for times in 1 2 3 4 5
    do
        rule_path=${data_dir}${data_name}"/rules.txt"
        train_data=${data_dir}${data_name}"/train/train.csv"
        test_data=${data_dir}${data_name}"/train/test.csv"
        output_dir=${data_dir}${data_name}"/train/models_baseline_bert/"
        python -u ${code_dir}use_transformers.py -data_name ${data_name} -rule_path ${rule_path} -train_data ${train_data} -test_data ${test_data} -output_dir ${output_dir} -times ${times} -cuda ${cuda}
    done
done
