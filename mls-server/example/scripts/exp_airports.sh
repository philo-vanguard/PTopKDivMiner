#!/bin/bash

echo -e "varying all for airports"

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



# ---------------------------------------------------------------- airports ----------------------------------------------------------------
dataID=1
w_supp=1
w_conf=1
w_rel_model=0.5
w_attr_non=1
w_pred_non=1
w_attr_dis=1
w_tuple_cov=1

# top-k-div, default setting
expOption="default"
./run_unit.sh ${dataID} ${expOption} ${suppDefault} ${confDefault} ${topKDefault} ${tupleNumDefault} ${numOfProcessorDefault} ${filterEnumNumber} ${relevanceDefualt} ${diversityDefualt} ${lambda} ${sampleRatioForTupleCov} ${w_supp} ${w_conf} ${w_attr_non} ${w_pred_non} ${w_attr_dis} ${w_tuple_cov} ${w_rel_model} ${max_X_len_default}

