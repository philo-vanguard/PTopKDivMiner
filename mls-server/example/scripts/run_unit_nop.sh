#!/bin/bash


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

exp=(
"vary_n"
"vary_supp"
"vary_conf"
"vary_k"
"vary_tuples"
"vary_topk"
"vary_synD"
"vary_synD_n"
)


dataID=$1

expOption=$2

supp=$3
conf=$4
topK=$5
tnum=$6
processor=$7

filterEnumNumber=$8

relevanceMeasures=$9
diversityMeasures=${10}
lambda=${11}
sampleRatioForTupleCov=${12}

w_supp=${13}
w_conf=${14}
w_attr_non=${15}
w_pred_non=${16}
w_attr_dis=${17}
w_tuple_cov=${18}
w_rel_model=${19}
MAX_X_LENGTH=${20}


predicateToIDFile="/tmp/rulefind/relevanceModel/"${task[${dataID}]}"/all_predicates.txt"
relevanceModelFile="/tmp/rulefind/relevanceModel/"${task[${dataID}]}"/model.txt"

if [ $dataID -eq 13 ] || [ $dataID -eq 14 ] || [ $dataID -eq 15 ] || [ $dataID -eq 16 ] || [ $dataID -eq 17 ]  # ncvoter, vary data size
then
    predicateToIDFile="/tmp/rulefind/relevanceModel/ncvoter/all_predicates.txt"
    relevanceModelFile="/tmp/rulefind/relevanceModel/ncvoter/model.txt"
fi


cd ..

resRootDir="./discoveryResultsTopKDiversified/"

mkdir -p ${resRootDir}

resDir=${resRootDir}${task[${dataID}]}"/filterEnumNumber"${filterEnumNumber}"_w_supp"${w_supp}"_conf"${w_conf}"_relModel"${w_rel_model}"_attrNon"${w_attr_non}"_predNon"${w_pred_non}"_attrDis"${w_attr_dis}"_tupleCov"${w_tuple_cov}"_lambda"${lambda}"/"

mkdir -p ${resDir}

echo -e "result output file: "${resDir}

tailFile="_len"${MAX_X_LENGTH}"_supp"${supp}"_conf"${conf}"_top"${topK}"_processor"${processor}"_rel_"${relevanceMeasures}"__div_"${diversityMeasures}".txt"

outputFile_ptopkminer_noLB_noUB='result_'${task[${dataID}]}"_noLB_noUB_"${expOption}${tailFile}



echo -e "---------- PTopkDivMiner-noLB-noUB algorithm ----------"

echo -e "output file name : "${outputFile_ptopkminer_noLB_noUB}

if [ -f ${resDir}${outputFile_ptopkminer_noLB_noUB} ] 
then 
    echo "The file exists: "${resDir}${outputFile_ptopkminer_noLB_noUB}
else
    ./run.sh  support=${supp} confidence=${conf} taskID=${task[${dataID}]} highSelectivityRatio=0 interestingness=1.5 skipEnum=false dataset=${task[${dataID}]} topK=${topK} round=1 maxTupleVariableNum=${tnum} ifPrune=false outputResultFile=${outputFile_ptopkminer_noLB_noUB} algOption="diversified" numOfProcessors=${processor} MLOption=1 ifClusterWorkunits=0 filterEnumNumber=${filterEnumNumber} lambda=${lambda} relevance=${relevanceMeasures} diversity=${diversityMeasures} sampleRatioForTupleCov=${sampleRatioForTupleCov} w_supp=${w_supp} w_conf=${w_conf} w_attr_non=${w_attr_non} w_pred_non=${w_pred_non} w_attr_dis=${w_attr_dis} w_tuple_cov=${w_tuple_cov} w_rel_model=${w_rel_model} predicateToIDFile=${predicateToIDFile} relevanceModelFile=${relevanceModelFile} MAX_X_LENGTH=${MAX_X_LENGTH}

    rm ${resDir}${outputFile_ptopkminer_noLB_noUB}
    hdfs dfs -get "/tmp/rulefind/"${task[${dataID}]}"/rule_all/"${outputFile_ptopkminer_noLB_noUB} ${resDir}
fi
