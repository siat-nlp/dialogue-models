# -*- coding: utf-8 -*-
import numpy as np
import json


PAD_ID = 0
UNK_ID = 1
GO_ID = 2
EOS_ID = 3
NONE_ID = 0
_START_VOCAB = ['_PAD', '_UNK', '_GO', '_EOS']


def load_knowledge(path):
    KB_tuple = {"csk_triples": [], "csk_entities": [], "kb_dict": []}
    with open('%s/resource.txt' % path) as f:
        d = json.loads(f.readline())
        KB_tuple["csk_triples"] = d['csk_triples']
        KB_tuple["csk_entities"] = d['csk_entities']
        KB_tuple["kb_dict"] = d['dict_csk']
        raw_vocab = d['vocab_dict']
    return raw_vocab, KB_tuple


def load_data(path, is_train=True):
    data_train, data_dev, data_test = [], [], []
    if is_train:
        with open('%s/trainset.txt' % path) as f:
            for idx, line in enumerate(f):
                if idx > 0 and idx % 100000 == 0:
                    print('read train file line %d' % idx)
                data_train.append(json.loads(line))
        with open('%s/validset.txt' % path) as f:
            for line in f:
                data_dev.append(json.loads(line))
    with open('%s/testset.txt' % path) as f:
        for line in f:
            data_test.append(json.loads(line))

    return data_train, data_dev, data_test


def build_vocab(path, raw_vocab, FLAGS, trans='transE'):
    print("Creating word vocabulary...")
    vocab_list = _START_VOCAB + sorted(raw_vocab, key=raw_vocab.get, reverse=True)
    if len(vocab_list) > FLAGS.vocab_size:
        vocab_list = vocab_list[:FLAGS.vocab_size]

    print("Creating entity vocabulary...")
    entity_list = ['_NONE', '_PAD_H', '_PAD_R', '_PAD_T', '_NAF_H', '_NAF_R', '_NAF_T']
    with open('%s/entity.txt' % path) as f:
        for i, line in enumerate(f):
            e = line.strip()
            entity_list.append(e)

    print("Creating relation vocabulary...")
    relation_list = []
    with open('%s/relation.txt' % path) as f:
        for i, line in enumerate(f):
            r = line.strip()
            relation_list.append(r)

    print("Loading word vectors...")
    vectors = {}
    with open('%s/glove.840B.300d.txt' % path) as f:
        for i, line in enumerate(f):
            s = line.strip()
            word = s[:s.find(' ')]
            vector = s[s.find(' ')+1:]
            vectors[word] = vector

    embed = []
    for word in vocab_list:
        if word in vectors:
            vector = list(map(float, vectors[word].split()))
        else:
            vector = np.zeros(FLAGS.embed_units, dtype=np.float32)
        embed.append(vector)
    embed = np.array(embed, dtype=np.float32)

    print("Loading entity vectors...")
    entity_embed = []
    with open('%s/entity_%s.txt' % (path, trans)) as f:
        for i, line in enumerate(f):
            s = line.strip().split('\t')
            entity_embed.append(list(map(float, s)))

    print("Loading relation vectors...")
    relation_embed = []
    with open('%s/relation_%s.txt' % (path, trans)) as f:
        for i, line in enumerate(f):
            s = line.strip().split('\t')
            relation_embed.append(s)

    entity_relation_embed = np.array(entity_embed+relation_embed, dtype=np.float32)

    return vocab_list, embed, entity_list, relation_list, entity_relation_embed


def gen_batched_data(data, KB_tuple, FLAGS):
    csk_triples = KB_tuple["csk_triples"]
    csk_entities = KB_tuple["csk_entities"]

    encoder_len = max([len(item['post']) for item in data]) + 1
    decoder_len = max([len(item['response']) for item in data]) + 1
    triple_num = max([len(item['all_triples']) for item in data]) + 1
    triple_len = max([len(tri) for item in data for tri in item['all_triples']])
    posts, responses, posts_length, responses_length = [], [], [], []
    entities, triples, matches, post_triples, response_triples = [], [], [], [], []
    match_triples, all_triples = [], []
    NAF = ['_NAF_H', '_NAF_R', '_NAF_T']

    def padding(sent, l):
        return sent + ['_EOS'] + ['_PAD'] * (l-len(sent)-1)

    def padding_triple(triple, num, l):
        new_triple = []
        triple = [[NAF]] + triple
        for tri in triple:
            new_triple.append(tri + [['_PAD_H', '_PAD_R', '_PAD_T']] * (l-len(tri)))
        pad_triple = [['_PAD_H', '_PAD_R', '_PAD_T']] * l
        return new_triple + [pad_triple] * (num-len(new_triple))

    for item in data:
        posts.append(padding(item['post'], encoder_len))
        responses.append(padding(item['response'], decoder_len))
        posts_length.append(len(item['post']) + 1)
        responses_length.append(len(item['response']) + 1)
        all_triples.append(padding_triple(
            [[csk_triples[x].split(', ') for x in triple] for triple in item['all_triples']],
            triple_num,
            triple_len))
        post_triples.append([[x] for x in item['post_triples']] + [[0]] * (encoder_len-len(item['post_triples'])))
        response_triples.append([NAF] +
                                [NAF if x == -1 else csk_triples[x].split(', ') for x in item['response_triples']] +
                                [NAF] * (decoder_len-1-len(item['response_triples'])))
        match_index = []
        for idx, x in enumerate(item['match_index']):
            _index = [-1] * triple_num
            if x[0] == -1 and x[1] == -1:
                match_index.append(_index)
            else:
                _index[x[0]] = x[1]
                t = all_triples[-1][x[0]][x[1]]
                assert (t == response_triples[-1][idx + 1])
                match_index.append(_index)
        match_triples.append(match_index + [[-1] * triple_num] * (decoder_len - len(match_index)))
        if not FLAGS.is_train:
            entity = [['_NONE'] * triple_len]
            for ent in item['all_entities']:
                entity.append([csk_entities[x] for x in ent] + ['_NONE'] * (triple_len - len(ent)))
            entities.append(entity + [['_NONE'] * triple_len] * (triple_num - len(entity)))

    batched_data = {'posts': np.array(posts),
                    'responses': np.array(responses),
                    'posts_length': posts_length,
                    'responses_length': responses_length,
                    'triples': np.array(all_triples),
                    'entities': np.array(entities),
                    'posts_triple': np.array(post_triples),
                    'responses_triple': np.array(response_triples),
                    'match_triples': np.array(match_triples)}
    return batched_data
