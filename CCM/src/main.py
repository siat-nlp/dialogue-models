# -*- coding: utf-8 -*-
import os
import time
import random
import json
import numpy as np
import tensorflow as tf
from nltk.translate.bleu_score import corpus_bleu
from .model import Model
import .utils as utils

random.seed(time.time())

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

tf.app.flags.DEFINE_string("data_dir", "", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "", "Training directory")
tf.app.flags.DEFINE_boolean("is_train", True, "Set to False to inference")
tf.app.flags.DEFINE_integer("vocab_size", 30000, "vocabulary size")
tf.app.flags.DEFINE_integer("num_entities", 21471, "entitiy vocabulary size")
tf.app.flags.DEFINE_integer("num_relations", 44, "relation size")
tf.app.flags.DEFINE_integer("embed_units", 300, "Size of word embedding")
tf.app.flags.DEFINE_integer("trans_units", 100, "Size of trans embedding")
tf.app.flags.DEFINE_integer("units", 512, "Size of each model layer")
tf.app.flags.DEFINE_integer("layers", 2, "Number of layers in the model")
tf.app.flags.DEFINE_integer("batch_size", 100, "Batch size to use during training")
tf.app.flags.DEFINE_integer("max_dec_lens", 60, "max tokens of decoder")
tf.app.flags.DEFINE_float("lr", 0.0001, "learning rate for training")
tf.app.flags.DEFINE_integer("per_checkpoint", 1000, "How many steps to do per checkpoint")
tf.app.flags.DEFINE_integer("inference_version", 0, "The version for inferencing")
tf.app.flags.DEFINE_boolean("log_parameters", True, "Set to True to show the parameters")
FLAGS = tf.app.flags.FLAGS


def model_train(model, sess, data_train, KB_tuple):
    batched_data = utils.gen_batched_data(data_train, KB_tuple, FLAGS)
    outputs = model.step_train(sess, batched_data)
    sentence_ppx = np.sum(outputs[0])
    decoder_loss = np.sum(outputs[1])
    return sentence_ppx, decoder_loss


def generate_summary(model, sess, data_train, KB_tuple):
    selected_data = [random.choice(data_train) for i in range(FLAGS.batch_size)]
    batched_data = utils.gen_batched_data(selected_data, KB_tuple, FLAGS)
    outputs = model.step_train(sess, batched_data, forward_only=True, summary=True)
    summary = outputs[-1]
    return summary


def model_evaluate(model, sess, data_dev, summary_writer, KB_tuple):
    ppx = np.zeros((1, ))
    st, ed, times = 0, FLAGS.batch_size, 0
    while st < len(data_dev):
        selected_data = data_dev[st:ed]
        batched_data = utils.gen_batched_data(selected_data, KB_tuple, FLAGS)
        outputs = model.step_train(sess, batched_data, forward_only=True)
        ppx += np.sum(outputs[0])
        st, ed = ed, ed+FLAGS.batch_size
        times += 1
    ppx /= len(data_dev)
    summary = tf.Summary()
    summary.value.add(tag='perplexity/dev', simple_value=np.exp(ppx))
    summary_writer.add_summary(summary, model.global_step.eval())
    print("   perplexity on dev set: %.2f" % np.exp(ppx))


def get_steps(train_dir):
    steps, metafiles, datafiles, indexfiles = [], [], [], []
    for root, dirs, files in os.walk(train_dir):
        if root == train_dir:
            filenames = files
            for filename in filenames:
                if 'meta' in filename:
                    metafiles.append(filename)
                if 'data' in filename:
                    datafiles.append(filename)
                if 'index' in filename:
                    indexfiles.append(filename)
    metafiles.sort()
    datafiles.sort()
    indexfiles.sort(reverse=True)
    for f in indexfiles:
        steps.append(int(f[11:-6]))

    return steps


def model_test(sess, saver, data_dev, KB_tuple, setnum=5000, max_step=800000):
    with open('%s/stopwords' % FLAGS.data_dir) as f:
        stopwords = json.loads(f.readline())
    steps = get_steps(FLAGS.train_dir)
    results = []

    with open('%s/test.res' % FLAGS.train_dir, 'w') as resfile, open('%s/test.log' % FLAGS.train_dir, 'w') as outfile:
        steps = [step for step in steps if step in range(max_step)]
        for step in steps:
            outfile.write('test for model-%d\n' % step)
            model_path = '%s/checkpoint-%08d' % (FLAGS.train_dir, step)
            print('restore from %s' % model_path)
            try:
                saver.restore(sess, model_path)
            except:
                continue
            st, ed = 0, FLAGS.batch_size
            loss = []
            while st < len(data_dev):
                selected_data = data_dev[st:ed]
                batched_data = utils.gen_batched_data(selected_data, KB_tuple, FLAGS)
                responses, ppx_loss = sess.run(['decoder_1/generation:0', 'decoder/ppx_loss:0'],
                                               {'enc_inps:0': batched_data['posts'],
                                                'enc_lens:0': batched_data['posts_length'],
                                                'dec_inps:0': batched_data['responses'],
                                                'dec_lens:0': batched_data['responses_length'],
                                                'entities:0': batched_data['entities'],
                                                'triples:0': batched_data['triples'],
                                                'match_triples:0': batched_data['match_triples'],
                                                'enc_triples:0': batched_data['posts_triple'],
                                                'dec_triples:0': batched_data['responses_triple']})
                loss += [x for x in ppx_loss]
                for response in responses:
                    result = []
                    for token in response:
                        token = str(token, encoding='utf-8')
                        if token != '_EOS':
                            result.append(token)
                        else:
                            break
                    results.append(result)
                st, ed = ed, ed+FLAGS.batch_size
            match_entity_sum = [.0] * 4
            cnt = 0
            posts = [data['post'] for data in data_dev]
            responses = [data['response'] for data in data_dev]
            match_triples = [data['match_triples'] for data in data_dev]
            all_triples = [data['all_triples'] for data in data_dev]
            all_entites = [data['all_entities'] for data in data_dev]

            hypotheses = []
            references = []

            for post, response, result, match_triple, triples, entities in \
                    zip(posts, responses, results, match_triples, all_triples, all_entites):
                setidx = int(cnt / setnum)
                result_matched_entities = []
                entities = [KB_tuple["csk_entities"][x] for entity in entities for x in entity]
                for word in result:
                    if word not in stopwords and word in entities:
                        result_matched_entities.append(word)

                post = " ".join([str(p) for p in post])
                response = [str(r) for r in response]
                references.append([response])
                response = " ".join(response)
                
                hypotheses.append(result)
                result = " ".join(result)

                outfile.write('post: %s\nresponse: %s\nresult: %s\nmatch_entity: %s\n\n' %
                              (post, response, result, " ".join(result_matched_entities)))
                match_entity_sum[setidx] += len(set(result_matched_entities))
                cnt += 1
            match_entity_sum = [m / setnum for m in match_entity_sum] + [sum(match_entity_sum) / len(data_dev)]
            losses = [np.sum(loss[x:x+setnum]) / float(setnum) for x in range(0, setnum*4, setnum)] + \
                     [np.sum(loss) / float(setnum*4)]
            losses = [np.exp(x) for x in losses]
            
            bleus_1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
            bleus_2 = corpus_bleu(references, hypotheses, weights=(0, 1, 0, 0))
            bleus_3 = corpus_bleu(references, hypotheses, weights=(0, 0, 1, 0))
            bleus_4 = corpus_bleu(references, hypotheses, weights=(0, 0, 0, 1))
            bleus_overall = corpus_bleu(references, hypotheses)
            bleus = [bleus_1, bleus_2, bleus_3, bleus_4, bleus_overall]

            def show(x):
                return ', '.join([str(v) for v in x])

            outfile.write('model: %d\n\tbleu: %s\n\tperplexity: %s\n\tmatch_entity_rate: %s\n%s\n\n' %
                          (step, bleus, show(losses), show(match_entity_sum), '=' * 50))
            resfile.write('model: %d\n\tbleu: %s\n\tperplexity: %s\n\tmatch_entity_rate: %s\n\n' %
                          (step, bleus, show(losses), show(match_entity_sum)))
            outfile.flush()
            resfile.flush()


def main():
    if FLAGS.train_dir[-1] == '/':
        FLAGS.train_dir = FLAGS.train_dir[:-1]
    if not os.path.exists(FLAGS.train_dir):
        os.mkdir(FLAGS.train_dir)
    if FLAGS.is_train:
        # load data sets
        data_train, data_dev, data_test = utils.load_data(FLAGS.data_dir, is_train=True)
        print("origin data train: ", len(data_train))

        # load KB & vocab
        raw_vocab, KB_tuple = utils.load_knowledge(FLAGS.data_dir)

        # build vocabs and knowledge graphs
        vocab, word_embed, entity_vocab, relation_vocab, \
        entity_relation_embed = utils.build_vocab(FLAGS.data_dir, raw_vocab, FLAGS)

        FLAGS.num_entities = len(entity_vocab)
        FLAGS.num_relations = len(relation_vocab)

        model = Model(word_embed, entity_relation_embed, vocab_size=FLAGS.vocab_size,
                      num_embed_units=FLAGS.embed_units, num_units=FLAGS.units, num_layers=FLAGS.layers,
                      num_entities=FLAGS.num_entities+FLAGS.num_relations, num_trans_units=FLAGS.trans_units,
                      max_length=FLAGS.max_dec_lens, learning_rate=FLAGS.lr)

        with tf.Session(config=config) as sess:
            if tf.train.get_checkpoint_state(FLAGS.train_dir):
                print("Reading model parameters from %s" % FLAGS.train_dir)
                model.saver.restore(sess, tf.train.latest_checkpoint(FLAGS.train_dir))
            else:
                print("Created model with fresh parameters.")
                tf.global_variables_initializer().run()
                sess = model.set_vocabs(sess, vocab, entity_vocab, relation_vocab)

            if FLAGS.log_parameters:
                model.print_parameters()

            summary_writer = tf.summary.FileWriter('%s/log' % FLAGS.train_dir, sess.graph)
            loss_step = np.zeros((1,))
            ppx_step = np.zeros((1,))
            train_len = len(data_train)
            print("Train data: ", train_len)

            while True:
                st, ed = 0, FLAGS.batch_size*FLAGS.per_checkpoint
                random.shuffle(data_train)
                while st < train_len:
                    start_time = time.time()
                    for batch in range(st, ed, FLAGS.batch_size):
                        # get batch train data
                        batch_train = data_train[batch:batch+FLAGS.batch_size]
                        # training model
                        sentence_ppx, decoder_loss = model_train(model, sess, batch_train, KB_tuple)
                        ppx_step += sentence_ppx / (ed - st)
                        loss_step += decoder_loss / (ed - st)

                    show = lambda a: '[%s]' % (' '.join(['%.2f' % x for x in a]))
                    print(" global step %d step-time %.2f loss %f ppx_loss %f perplexity %s"
                          % (model.global_step.eval(),
                             (time.time() - start_time) / ((ed - st) / FLAGS.batch_size), loss_step, ppx_step,
                             show(np.exp(ppx_step))))
                    # save the model per training step
                    model.saver.save(sess, '%s/checkpoint' % FLAGS.train_dir, global_step=model.global_step)
                    summary = tf.Summary()
                    summary.value.add(tag='decoder_loss/train', simple_value=loss_step)
                    summary.value.add(tag='perplexity/train', simple_value=np.exp(ppx_step))
                    summary_writer.add_summary(summary, model.global_step.eval())
                    summary_model = generate_summary(model, sess, data_train, KB_tuple)
                    summary_writer.add_summary(summary_model, model.global_step.eval())

                    # evaluate model on dev set
                    model_evaluate(model, sess, data_dev, summary_writer, KB_tuple)
                    loss_step = np.zeros((1,))
                    ppx_step = np.zeros((1,))
                    st, ed = ed, min(train_len, ed + FLAGS.batch_size * FLAGS.per_checkpoint)
                # save the model per epoch
                model.saver_epoch.save(sess, '%s/epoch/checkpoint' % FLAGS.train_dir, global_step=model.global_step)
    else:
        # load test test
        _, _, data_test = utils.load_data(FLAGS.data_dir, is_train=False)
        # load KB & vocab
        raw_vocab, KB_tuple = utils.load_knowledge(FLAGS.data_dir)

        model = Model(word_embed=None, entity_embed=None, vocab_size=FLAGS.vocab_size,
                      num_embed_units=FLAGS.embed_units, num_units=FLAGS.units, num_layers=FLAGS.layers,
                      num_entities=FLAGS.num_entities+FLAGS.num_relations, num_trans_units=FLAGS.trans_units,
                      max_length=FLAGS.max_dec_lens, learning_rate=FLAGS.lr)

        with tf.Session(config=config) as sess:
            if FLAGS.inference_version == 0:
                model_path = tf.train.latest_checkpoint(FLAGS.train_dir)
            else:
                model_path = '%s/checkpoint-%08d' % (FLAGS.train_dir, FLAGS.inference_version)
            print('restore from %s' % model_path)
            model.saver.restore(sess, model_path)
            saver = model.saver
            # test model on test set
            model_test(sess, saver, data_test, KB_tuple)


if __name__ == '__main__':
    main()
