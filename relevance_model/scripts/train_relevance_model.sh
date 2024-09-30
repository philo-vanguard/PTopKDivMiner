#!/bin/bash

dir=$1
data_dir=${dir}"relevance_model/datasets/"
code_dir=${dir}"relevance_model/"

for data_name in "airports" "hospital" "inspection" "ncvoter" "dblp" "adults"
do
    for times in 1 2 3 4 5
    do
        lr=0.001
        hidden_size=100
        rees_embed_dim=100
        epochs=400
        batch_size=128
        rules_file=${data_dir}${data_name}"/rules.txt"
        train_file=${data_dir}${data_name}"/train/train.csv"
        test_file=${data_dir}${data_name}"/train/test.csv"
        pretrained_matrix_file=${data_dir}${data_name}"/predicateEmbedds.csv"
        output_model_dir=${data_dir}${data_name}"/train/model/"
        mkdir -p ${output_model_dir}
        all_predicates_file=${data_dir}${data_name}"/all_predicates.txt"

        python -u ${code_dir}relevanceFixedEmbeds_main.py -rules_file ${rules_file} -train_file ${train_file} -test_file ${test_file} -pretrained_matrix_file ${pretrained_matrix_file} -output_model_dir ${output_model_dir} -all_predicates_file ${all_predicates_file} -lr ${lr} -hidden_size ${hidden_size} -rees_embed_dim ${rees_embed_dim} -epochs ${epochs} -batch_size ${batch_size} -times ${times} -withBertInitialization 0 
    done
done
