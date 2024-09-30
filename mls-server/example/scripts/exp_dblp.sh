#!/bin/bash

echo -e "varying all for aminer"

task=(
"adults"
"airports"
"flight"
"hospital"
"inspection"
"ncvoter"
"aminer"
"tax100w"
"tax200w"
"tax400w"
"tax600w"
"tax800w"
"tax1000w"
"ncvoter_0.2"
"ncvoter_0.4"
"ncvoter_0.6"
"ncvoter_0.8"
"ncvoter_1.0"
)

relevanceMeasures=(
"support"
"confidence"
"relevance_model"
)

diversityMeasures=(
"attribute_nonoverlap"
"predicate_nonoverlap"
"attribute_distance"
"tuple_coverage"
)


suppDefault=0.000001
confDefault=0.75
topKDefault=10
tupleNumDefault=2
numOfProcessorDefault=20
filterEnumNumber=10
max_X_len_default=5

lambda=2.0
sampleRatioForTupleCov=0.0001
# relevanceDefualt="support#confidence#relevance_model" # split by '#'
relevanceDefualt="relevance_model"
diversityDefualt="attribute_nonoverlap#predicate_nonoverlap#attribute_distance#tuple_coverage" # split by '#'



# ---------------------------------------------------------------- aminer ----------------------------------------------------------------
dataID=6
w_supp=2
w_conf=1
w_rel_model=0.5
w_attr_non=2
w_pred_non=2
w_attr_dis=1
w_tuple_cov=1.5



expOption="vary_conf"
if [ ${expOption} = "vary_conf" ]
then
for conf in 0.95 0.9 0.85 0.8 0.75
do
    echo -e "supp = "${suppDefault}" and conf = "${conf}" with data "${task[${dataID}]}", relevance: "${relevanceDefualt}", diversity: "${diversityDefualt}
    # top-k-div
    ./run_unit.sh ${dataID} ${expOption} ${suppDefault} ${conf} ${topKDefault} ${tupleNumDefault} ${numOfProcessorDefault} ${filterEnumNumber} ${relevanceDefualt} ${diversityDefualt} ${lambda} ${sampleRatioForTupleCov} ${w_supp} ${w_conf} ${w_attr_non} ${w_pred_non} ${w_attr_dis} ${w_tuple_cov} ${w_rel_model} ${max_X_len_default}
    
    # top-k-div-nop
    ./run_unit_nop.sh ${dataID} ${expOption} ${suppDefault} ${conf} ${topKDefault} ${tupleNumDefault} ${numOfProcessorDefault} ${filterEnumNumber} ${relevanceDefualt} ${diversityDefualt} ${lambda} ${sampleRatioForTupleCov} ${w_supp} ${w_conf} ${w_attr_non} ${w_pred_non} ${w_attr_dis} ${w_tuple_cov} ${w_rel_model} ${max_X_len_default}

    # top-k-div-NoDiv
    diversity="null"
    ./run_unit.sh ${dataID} ${expOption} ${suppDefault} ${conf} ${topKDefault} ${tupleNumDefault} ${numOfProcessorDefault} ${filterEnumNumber} ${relevanceDefualt} ${diversity} ${lambda} ${sampleRatioForTupleCov} ${w_supp} ${w_conf} ${w_attr_non} ${w_pred_non} ${w_attr_dis} ${w_tuple_cov} ${w_rel_model} ${max_X_len_default}

    # top-k-div-NoRel
    relevance="null"
    ./run_unit.sh ${dataID} ${expOption} ${suppDefault} ${conf} ${topKDefault} ${tupleNumDefault} ${numOfProcessorDefault} ${filterEnumNumber} ${relevance} ${diversityDefualt} ${lambda} ${sampleRatioForTupleCov} ${w_supp} ${w_conf} ${w_attr_non} ${w_pred_non} ${w_attr_dis} ${w_tuple_cov} ${w_rel_model} ${max_X_len_default}

done
fi

