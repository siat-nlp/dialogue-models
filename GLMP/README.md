GLMP
=============================

This is a pytorch implementation of the model **GLMP** [[Global-to-local Memory Pointer Networks for Task-Oriented Dialogue](https://arxiv.org/pdf/1901.04713.pdf)] (ICLR-2019).

## Requirements

* Python>=3.6
* PyTorch>=1.0
* numpy
* nltk
* scikit-learn

## Dataset

The dataset can be downloaded from [2019 Language and Intelligence Challenge](http://lic2019.ccf.org.cn/talk), we just put example train data under the folder ```./data/resource```.

## Quickstart

### Step 1: Preprocess the data

Put the data provided by the organizer under the data folder and rename them  train/dev/test.txt: 
```
./data/resource/train.txt
./data/resource/dev.txt
./data/resource/test.txt
```

Process the data with the following commands:
```
sh process_data.sh
```

### Step 2: Train the model

Train model with the following commands:

```
sh run_train.sh
```

### Step 3: Test & evaluate the model

Test & evaluate model with the following commands:

```
sh run_test.sh
sh run_eval.sh
```

## References

[1] https://github.com/jasonwu0731/GLMP \
[2] https://github.com/baidu/knowledge-driven-dialogue/tree/master/generative_pt