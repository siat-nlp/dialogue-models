#!/bin/bash

# set python path according to your actual environment
pythonpath='python3'

# set data path
datapath=./data

# DATA_TYPE = "train" or "dev"
datatype=(train dev)

# data preprocessing
for ((i=0; i<${#datatype[*]}; i++))
do
    corpus_file=${datapath}/resource/${datatype[$i]}.txt
    sample_file=${datapath}/resource/sample.${datatype[$i]}.txt
    text_file=${datapath}/json.${datatype[$i]}.txt
    vocab_file=${datapath}/${datatype[$i]}.vocab

    # step 1: firstly have to convert session data to sample data
    ${pythonpath} ./src/scripts/convert_session_to_sample.py ${corpus_file} ${sample_file}

    # step 2: convert sample data to text data required by the model
    ${pythonpath} ./src/scripts/convert_sample_to_json.py ${sample_file} ${text_file} ${vocab_file}
done
