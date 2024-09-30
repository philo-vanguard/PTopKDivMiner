# Top-k Diversified REE Discovery
This paper studies the problem of discovering top-𝑘 relevant and diversified rules. Given a real-life dataset, it is to mine a set of 𝑘 rules that are as close to users’ interest as possible, and meanwhile, as diverse to each other as possible. It aims to reduce excessive irrelevant rules commonly returned by rule discovery. As a testbed, we consider Entity Enhancing Rules (REEs), which subsume popular data quality rules as special cases. We train a relevance model to learn users’ prior knowledge, rank rules based on users’ need, and propose four diversity measures to assess the diversity between rules. Based on these measures, we formulate a new discovery problem. We show that the bi-criteria discovery problem is NP-complete and hard to approximate. This said, we develop a practical algorithm for the problem, and prove its approximation bounds under certain conditions. Moreover, we develop optimization techniques to speed up the process, and parallelize the algorithm such that it guarantees to reduce runtime when given more processors.

For more details, see our paper:
> Wenfei Fan, Ziyan Han, Min Xie, and Guangyi Zhang. [*Discovering Top-k Relevant and Diversified Rules*](https://philo-vanguard.github.io/files/papers/Rule-Discovery-Top-k-Diversified-SIGMOD25.pdf). In SIGMOD (2025). ACM.

<br>

The codes consist of two main parts:
1. (Step 1) relevance_model: codes for learning relevance model M_rel;  
2. (Step 2) mls-server: codes for discovering top-k relevant and diversified rules;  

The relevance model, trained in step 1, plays a crucial role in the subsequent rule discovery phase (step 2). It enables the system to effectively identify and discover valuable rules.

## Installation
Before building the projects, the following prerequisites need to be installed:
* Java JDK 1.8
* Maven
* Transformers
* Tensorflow 2.6.2
* Pytorch 1.10.2
* Huggingface distilbert-base-uncased

## Datasets and Models
All datasets and well-trained relevance models used in this paper have been uploaded in Google Drive[https://drive.google.com/drive/folders/1iEZAzt6xj8K-A-8oK5XMkLBLn-Vtb_SV?usp=sharing].
You can download them, skip step 1, and run step 2 directly.

Below, we show the process of learning relevance model (step 1) and running rule discovery (step 2),
using 'airports' dataset as an example.

## Step 1: relevance model
The *relevance_model* folder contains the datasets and source code for learning relevance model M_rel.

### 1. Rules and data used for training relevance model
```
ls ./relevance_model/datasets/airports/
airports.csv  airports_sample.csv  airports_sample_clean.csv  all_predicates.txt  rules.txt  predicateEmbedds.csv  train/

ls ./relevance_model/datasets/airports/train/
train.csv test.csv
```

The file 'airports.csv' contains the raw data, while 'airports_sample.csv' is a sample of 'airports.csv', referred to as D_dirty in the paper. On the other hand, 'airports_sample_clean.csv' is a cleaned version of 'airports_sample.csv', denoted as D_clean in the paper.

The file 'rules.txt' is a set of rules used in the paper for relevance model training,
discovered from D_dirty, i.e., airports_sample.csv.

The file 'predicateEmbedds.csv' contains the embeddings for each predicate in 'all_predicates.txt',
learned by [Pytorch-Biggraph](https://github.com/facebookresearch/PyTorch-BigGraph).

The folder 'train/' contains the training and testing data for relevance model.

### 1) How to generate data for Pytorch-Biggraph to obtain the predicate embeddings
```
cd ./relevance_model/preprocess/construct_data_PytorchBiggraph/
python construct_triples.py
python construct_data.py
```
The results will be saved in './relevance_model/datasets/airports/train_pytorchBiggraph/',
which can be used to gerenerate the embeddings of predicates (as relations) by running codes in [Pytorch-Biggraph](https://github.com/facebookresearch/PyTorch-BigGraph).
(You may refer to the script in ./relevance_model/scripts/run_pytorch_biggraph.sh)

### 2) How to construct training and testing instances for relevance model
```
cd ./relevance_model/preprocess/
python construct_labelled_data_tupleError.py
```
The results will be saved in './relevance_model/datasets/airports/train/',
containing the training and testing data for relevance model.


### 2. Train the relevance model

### 1) Our model
```
cd ./relevance_model/scripts/
./train_relevance_model.sh ${dir}
```
Here the arguments **dir** is the absolute path of relevance_model in the local settings.

The results will be saved in './relevance_model/datasets/airports/train/model/'

### 2) Other baselines
```
cd ./relevance_model/scripts/
./train_relevance_model_initializedByBert.sh ${dir}       # for a variant of M_rel initialized by Bert
./train_baseline_bert.sh ${dir} ${cuda}                   # for Bert baseline
./train_baseline_subjective_model.sh ${dir}               # for subjective model M_sub
```
where **dir** is the absolute path of relevance_model in the local settings, and **cuda** refers to the gpu id.

The results will be saved in './relevance_model/datasets/airports/train/model_initializedByBert/',
'./relevance_model/datasets/airports/train/models_baseline_bert/', and 
'./relevance_model/datasets/airports/train/models_baseline_subjective/', respectively.

## Step 2: rule discovery
The *mls-server* folder contains the source code for top-k relevant and diversified REE discovery.
Below we give a toy example.

### 1. Put the datasets into HDFS:
```
hdfs dfs -mkdir /tmp/diversified_data/
hdfs dfs -put airports.csv /tmp/diversified_data/
```

### 2. Put the files related to relevance model, from ./relevance_model/datasets/, into HDFS:
```
hdfs dfs mkdir -p /tmp/rulefind/relevanceModel/airports/
hdfs dfs -put ./relevance_model/datasets/airports/all_predicates.txt /tmp/rulefind/relevanceModel/airports/
hdfs dfs -put ./relevance_model/datasets/airports/train/model/model.txt /tmp/rulefind/relevanceModel/airports/
```

### 3. Download all the dependencies (https://drive.google.com/drive/folders/1Gviqt7zcaRGQho4x5i6hPnuwPmWonWFR?usp=sharing), then move the directory lib/ into mls-server/example/:
```
cd mls-server/
mv lib/ example/
```

### 4. Compile and build the project:
```
mvn package
```
Then move and replace the **mls-server-0.1.1.jar** from mls-server/target/ to example/lib/:
```
mv target/mls_server-0.1.1.jar example/lib/
```

### 5. After all these preparation, run the toy example:
```
cd example/scripts/
./exp_airports.sh
```
The results, i.e., discovered rules will be shown in discoveryResultsTopKDiversified/, as 'resRootDir' in run_unit.sh shows.


<font color=red> 
Notice that, if you want to reproduce all the experiments in the paper, you may run the 'reproduce_all_experiments.sh' file, as follows.
</font>

```
cd example/scripts/
./reproduce_all_experiments.sh
```

### 6. Explanations for the parameters in scripts
* dataID: the ID of dataset;
* expOption: the description of discovery task;
* suppDefault: the threshold for support;
* confDefault: the threshold for confidence;
* topKDefault: the k size in discovery;
* tupleNumDefault: the number of tuple variables;
* numOfProcessorDefault: the number of processors;
* relevanceDefualt: the relevance metrics used in discovery;
* diversityDefualt: the diversity metrics used in discovery;
* lambda: the weight of diversity score;
* sampleRatioForTupleCov: the sample ratio for computing tuple coverage;
* w_supp: the weight of support metric when computing score;
* w_conf: the weight of confidence metric when computing score;
* w_attr_non: the weight of attribute nonoverlap when computing score;
* w_pred_non: the weight of predicate nonoverlap when computing score;
* w_attr_dis: the weight of attribute distance when computing score;
* w_tuple_cov: the weight of tuple coverage when computing score;
* w_rel_model: the weight of relevance model when computing score;
* max_X_len_default: the max length of X of rules;


## Release License
Please see the **LICENSE.txt** file.

