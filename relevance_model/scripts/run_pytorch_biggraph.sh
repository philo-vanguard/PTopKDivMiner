#!/bin/bash

for data in airports hospital inspection ncvoter dblp adults
do

    torchbiggraph_import_from_tsv \
        --lhs-col=0 --rel-col=1 --rhs-col=2 \
        torchbiggraph/examples/configs/${data}.py \
        data/${data}/train.csv \
        data/${data}/valid.csv \
        data/${data}/test.csv

    torchbiggraph_train \
        torchbiggraph/examples/configs/${data}.py \
        -p edge_paths=data/${data}/train_partitioned

    torchbiggraph_eval \
        torchbiggraph/examples/configs/${data}.py \
        -p edge_paths=data/${data}/test_partitioned \
        -p relations.0.all_negs=true \
        -p num_uniform_negs=0

    mkdir -p embeddings/${data}/

    torchbiggraph_export_to_tsv \
        torchbiggraph/examples/configs/${data}.py \
        --entities-output embeddings/${data}/entity_embeddings.tsv \
        --relation-types-output embeddings/${data}/relation_types_parameters.tsv

done
