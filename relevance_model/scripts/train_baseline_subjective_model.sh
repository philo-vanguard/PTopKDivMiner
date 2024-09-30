#!/bin/bash

dir=$1
data_dir=${dir}"relevance_model/datasets/"
code_dir=${dir}"relevance_model/baseline/subjective_model/"

for data_name in "airports" "hospital" "inspection" "ncvoter" "dblp" "adults"
do
    for times in 1 2 3 4 5
    do
        lr=0.001
        token_embed_dim=100
        hidden_size=100
        rees_embed_dim=100
        epochs=400
        batch_size=128
        rules_file=${data_dir}${data_name}"/rules.txt"
        train_file=${data_dir}${data_name}"/train/train.csv"
        test_file=${data_dir}${data_name}"/train/test.csv"
        output_model_dir=${data_dir}${data_name}"/train/models_baseline_subjective/"
        mkdir -p ${output_model_dir}
        model_file=${output_model_dir}"interestingness_model"
        model_txt_file=${output_model_dir}"interestingness_model"
        vobs_file=${output_model_dir}"vobs.txt"
        all_predicates_file=${data_dir}${data_name}"/all_predicates.txt"
        modelOpt="distilbert-base-uncased"

        python -u ${code_dir}interestingnessFixedEmbeds_main.py -rules_file ${rules_file} -train_file ${train_file} -test_file ${test_file} -output_model_dir ${output_model_dir} -model_file ${model_file} -model_txt_file ${model_txt_file} -vobs_file ${vobs_file} -all_predicates_file ${all_predicates_file} -modelOpt ${modelOpt} -lr ${lr} -token_embed_dim ${token_embed_dim} -hidden_size ${hidden_size} -rees_embed_dim ${rees_embed_dim} -epochs ${epochs} -batch_size ${batch_size} -times ${times}
    done
done
