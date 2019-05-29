import json
import collections

from src.inputters.dataset import UNK_token, PAD_token, EOS_token, SOS_token, TOPIC_A_PTR, TOPIC_B_PTR
from src.inputters.dataset import LicDataset
from torch.utils.data import DataLoader


TOKEN_OFFSET = 6


def prepare_batcher(data_file, word2index, batch_size=64, is_shuffle=False):
    data = []
    with open(data_file, 'r') as fr:
        for i, line in enumerate(fr):
            context = json.loads(line.strip(), encoding="utf-8", object_pairs_hook=collections.OrderedDict)
            data.append(context)

    data_info = {}
    for k in data[0].keys():
        data_info[k] = []
    for pair in data:
        for k in pair.keys():
            data_info[k].append(pair[k])

    dataset = LicDataset(data_info, word2index, word2index)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=is_shuffle,
                             collate_fn=dataset.collate_fn)
    return data_loader


def load_vocab(vocab_file, vocab_size):
    word2index = {"PAD": PAD_token,
                  "SOS": SOS_token,
                  "EOS": EOS_token,
                  "UNK": UNK_token,
                  "@topic_a": TOPIC_A_PTR,
                  "@topic_b": TOPIC_B_PTR}
    index2word = dict([(v, k) for k, v in word2index.items()])

    with open(vocab_file, 'r') as fr:
        for i, line in enumerate(fr):
            if i == vocab_size:
                break
            word = line.split('\t')[0]
            index2word[i+TOKEN_OFFSET] = word
            word2index[word] = i + TOKEN_OFFSET
    print("Total vocabs:", len(word2index))

    return word2index, index2word
