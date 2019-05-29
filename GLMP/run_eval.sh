#!/bin/bash

# set python path according to your actual environment
pythonpath='python3'

# set data path
datapath=./data

# DATA_TYPE = "dev" or "test"
datapart=dev
sample_file=${datapath}/resource/sample.${datapart}.txt

# if you eval dev.txt, you can run the following command to get result
if [ "${datapart}"x != "test"x ]; then
    ${pythonpath} ./src/scripts/convert_result_for_eval.py ${sample_file} ./output/test.result.final ./output/test.result.eval
    ${pythonpath} ./src/scripts/eval.py ./output/test.result.eval
fi
