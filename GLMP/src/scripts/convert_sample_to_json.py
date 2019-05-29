import sys
import json
import collections

MEM_TOKEN_SIZE = 4


def build_vocab(tokens, vocab_count):
    token_list = tokens.split()
    for tok in token_list:
        if tok in vocab_count.keys():
            vocab_count[tok] += 1
        else:
            vocab_count[tok] = 1


def process_one_conversation(text, vocab_count):
    conversation = json.loads(text.strip(), encoding="utf-8", object_pairs_hook=collections.OrderedDict)
    goal = conversation["goal"]
    knowledge = conversation["knowledge"]
    history = conversation["history"] if len(conversation["history"]) > 0 else ["null"]
    response = conversation["response"] if "response" in conversation else "null"

    video_entities, person_entities = [], []
    context_arr, conv_arr, kb_arr = [], [], []
    all_entities = {'topic_a': [],
                    'topic_b': []
                    }
    topic_a = goal[0][1]
    topic_b = goal[0][2]

    nid = 0
    for i, triple in enumerate(knowledge):
        [s, p, o] = triple
        triple_str = " ".join(triple)
        build_vocab(triple_str, vocab_count)

        assert s in [topic_a, topic_b]
        o_tokens = o.split()
        if s == topic_a:
            all_entities['topic_a'].append(s)
            for tok in o_tokens:
                all_entities['topic_a'].append(tok)
        else:
            all_entities['topic_b'].append(s)
            for tok in o_tokens:
                all_entities['topic_b'].append(tok)
        if u"领域" == p:
            if topic_a == s:
                domain_a = o
                if domain_a == u"电影":
                    video_entities.append(topic_a)
                else:
                    person_entities.append(topic_a)
            elif topic_b == s:
                domain_b = o
                if domain_b == u"电影":
                    video_entities.append(topic_b)
                else:
                    person_entities.append(topic_b)

        kb_info = generate_memory(triple, "", str(nid))
        kb_arr += kb_info
        context_arr = kb_info + context_arr

    for i, utterance in enumerate(history):
        if utterance == 'null':
            gen_m = generate_memory(utterance, "$u", 0)
        elif i % 2 == 0:
            build_vocab(utterance, vocab_count)

            nid += 1
            gen_m = generate_memory(utterance, "$s", str(nid))
        else:
            build_vocab(utterance, vocab_count)
            gen_m = generate_memory(utterance, "$u", str(nid))
        context_arr += gen_m
        conv_arr += gen_m

    build_vocab(response, vocab_count)

    # get gold entity for each response
    gold_ent = []
    for w in response.split():
        if w in all_entities['topic_a'] or w in all_entities['topic_b']:
            gold_ent.append(w)

    # get local pointer position for each word in system response
    ptr_index = []
    for key in response.split():
        index = [loc for loc, val in enumerate(context_arr) if (val[0] == key and key in gold_ent)]
        if len(index) > 0:
            index = max(index)
        else:
            index = len(context_arr)
        ptr_index.append(index)

    # get global pointer labels for words in system response, the 1 in the end is for the NULL token
    selector_index = [1 if (word_arr[0] in gold_ent or word_arr[0] in response.split())
                      else 0 for word_arr in context_arr] + [1]

    # get sketch response
    topic_entity = [topic_a, topic_b]
    sketch_response = generate_template(topic_entity, response)

    data_detail = {
        'context_arr': list(context_arr + [['$$$$'] * MEM_TOKEN_SIZE]),  # $$$$ is NULL token
        'response': response,
        'sketch_response': sketch_response,
        'ptr_index': ptr_index + [len(context_arr)],
        'selector_index': selector_index,
        'ent_index': gold_ent,
        'conv_arr': list(conv_arr),
        'kb_arr': list(kb_arr)
    }
    return data_detail


def generate_memory(sent, speaker, time):
    sent_new = []

    if speaker == "$u" or speaker == "$s":  # dialogue memory
        sent_token = sent.split(' ')
        for idx, word in enumerate(sent_token):
            temp = [word, speaker, 'turn'+str(time), 'word'+str(idx)] + ["PAD"]*(MEM_TOKEN_SIZE-4)
            sent_new.append(temp)
    else:  # knowledge memory
        sent_token = sent[::-1] + ["PAD"]*(MEM_TOKEN_SIZE-len(sent))
        sent_new.append(sent_token)
    return sent_new


def generate_template(topic_entity, sentence):
    """
    Based on the system response and the provided entity table, the output is the sketch response.
    """
    sketch_response = []
    for word in sentence.split():
        if word not in topic_entity:
            sketch_response.append(word)
        else:
            if word == topic_entity[0]:
                ent_type = 'topic_a'
            else:
                ent_type = 'topic_b'
            sketch_response.append('@' + ent_type)

    sketch_response = " ".join(sketch_response)
    return sketch_response


def convert_sample_to_json(sample_file, json_file, vocab_file=None):
    print("Reading lines from %s" % sample_file)

    vocab_count = {}
    with open(sample_file, 'r') as fr, open(json_file, 'w') as fw:
        for i, line in enumerate(fr):
            text_dict = process_one_conversation(line, vocab_count)
            text_json = json.dumps(text_dict, ensure_ascii=False)
            fw.write(text_json + "\n")
            if i > 0 and i % 10000 == 0:
                print("line %d done" % i)

    if vocab_file is not None:
        print("Building vocabs...")
        vocab_sorted = sorted(vocab_count.items(), key=lambda tup: tup[1], reverse=True)
        with open(vocab_file, 'w') as fw:
            for word, freq in vocab_sorted:
                fw.write(word + '\t' + str(freq) + '\n')


if __name__ == '__main__':
    try:
        convert_sample_to_json(sys.argv[1], sys.argv[2], sys.argv[3])
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")