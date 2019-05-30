#!/bin/bash
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################

# set gpu id to use
export CUDA_VISIBLE_DEVICES=2

# generalizes target_a/target_b of goal for all outputs, replaces them with slot mark
TOPIC_GENERALIZATION=1

# set python path according to your actual environment
pythonpath='python3'

# the prefix of the file name used by the model, must be consistent with the configuration in network.py
prefix=lic

# put all data set that used and generated for testing under this folder: datapath
# for more details, please refer to the following data processing instructions
datapath=./data
output_dir=./output

# in test stage, you can eval dev.txt or test.txt
# the "dev.txt" and "test.txt" are the original data provided by the organizer and
# need to be placed in this folder: datapath/resource/
# the following preprocessing will generate the actual data needed for model testing
# after testing dev.txt, you can run eval.py to get the final eval score,
# because dev.txt is session data, you have all the utterances both of bot and user
# after testing test.txt, you can upload the predict to the competition website to get result
# DATA_TYPE = "dev" or "test"
datapart=test

# ensure that each file is in the correct path
#     1. put the data provided by the organizers under this folder: datapath/resource/
#            - the data provided consists of three parts: train.txt dev.txt test.txt
#            - the train.txt and dev.txt are session data, the test.txt is sample data
#            - in test stage, we just use the dev.txt or test.txt
#     2. the sample data extracted from session data is in this folder: datapath/resource/
#     3. the text file required by the model is in this folder: datapath
#     4. the topic file used to generalize data is in this directory: datapath
corpus_file=${datapath}/resource/${datapart}.txt
sample_file=${datapath}/resource/sample.${datapart}.txt
text_file=${datapath}/${prefix}.test
topic_file=${datapath}/${prefix}.test.topic

# step 1: if eval dev.txt, firstly have to convert session data to sample data
# if eval test.txt, we can use test.txt provided by the organizer directly.
if [ "${datapart}"x = "test"x ]; then
    sample_file=${corpus_file}
else
    ${pythonpath} ./tools/convert_session_to_sample.py ${corpus_file} ${sample_file}
fi

# step 2: convert sample data to text data required by the model
${pythonpath} ./tools/convert_conversation_corpus_to_model_text.py ${sample_file} ${text_file} ${topic_file} ${TOPIC_GENERALIZATION}

# step 3: predict by model
ckpt_prefix=best.model
mkdir -p ${output_dir}/${ckpt_prefix}

${pythonpath} ./main.py --test --ckpt models/${ckpt_prefix} --gen_file ${output_dir}/${ckpt_prefix}/${datapart}.result --use_posterior False

# step 4: replace slot mark generated during topic generalization with real text
${pythonpath} ./tools/topic_materialization.py ${output_dir}/${ckpt_prefix}/${datapart}.result ${output_dir}/${ckpt_prefix}/${datapart}.result.final ${topic_file}

# step 5: if you eval dev.txt, you can run the following command to get result
# if you eval test.txt, you can upload the ./output/test.result.final to the competition website to get result
if [ "${datapart}"x != "test"x ]; then
    ${pythonpath} ./tools/convert_result_for_eval.py ${sample_file} ${output_dir}/${ckpt_prefix}/${datapart}.result.final ${output_dir}/${ckpt_prefix}/${datapart}.result.eval
    ${pythonpath} ./tools/eval.py ${output_dir}/${ckpt_prefix}/${datapart}.result.eval ${output_dir}/${ckpt_prefix}/${datapart}.eval.txt
fi

