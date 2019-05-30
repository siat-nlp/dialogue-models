CCM
=============================

This is a tensorflow implementation of the model **CCM** [[Commonsense Knowledge Aware Conversation Generation with Graph Attention](https://www.ijcai.org/proceedings/2018/0643.pdf)] (IJCAI-2018).

## Requirements

* Python>=3.6
* Tensorflow>=1.5
* numpy
* nltk

## Dataset

The commonsense conversation dataset can be downloaded from [here](http://coai.cs.tsinghua.edu.cn/file/commonsense_conversation_dataset.tar.gz), which are put under the folder ```./data```.

## Quickstart

* Train the model
  ```
  sh run_train.sh
  ```
* Test the model
  ```
  sh run_test.sh
  ```


## References

[1] https://github.com/tuxchow/ccm