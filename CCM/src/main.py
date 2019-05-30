# -*- coding: utf-8 -*-
import os
import time
import random
import json
import numpy as np
import tensorflow as tf
from .utils import load_data, load_knowledge, build_vocab
from .model import Model
from .trainer import model_train, model_evaluate, model_test, generate_summary

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
tf.app.flags.DEFINE_integer("per_checkpoint", 100, "How many steps to do per checkpoint")
tf.app.flags.DEFINE_integer("inference_version", 0, "The version for inferencing")
tf.app.flags.DEFINE_boolean("log_parameters", True, "Set to True to show the parameters")
FLAGS = tf.app.flags.FLAGS


def main():
    if FLAGS.train_dir[-1] == '/':
        FLAGS.train_dir = FLAGS.train_dir[:-1]
    if not os.path.exists(FLAGS.train_dir):
        os.mkdir(FLAGS.train_dir)
    if FLAGS.is_train:
        # load data sets
        data_train, data_dev, data_test = load_data(FLAGS.data_dir, is_train=True)
        print("data train: ", len(data_train))

        # load KB & vocab
        raw_vocab, KB_tuple = load_knowledge(FLAGS.data_dir)

        # build vocabs and knowledge graphs
        vocab, word_embed, entity_vocab, relation_vocab, \
        entity_relation_embed = build_vocab(FLAGS.data_dir, raw_vocab, FLAGS)

        FLAGS.num_entities = len(entity_vocab)
        FLAGS.num_relations = len(relation_vocab)

        model = Model(word_embed, entity_relation_embed, vocab_size=FLAGS.vocab_size,
                      num_embed_units=FLAGS.embed_units, num_units=FLAGS.units, num_layers=FLAGS.layers,
                      num_entities=FLAGS.num_entities+FLAGS.num_relations, num_trans_units=FLAGS.trans_units,
                      max_length=FLAGS.max_dec_lens, learning_rate=FLAGS.lr)
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=3,
                               pad_step_number=True)
        saver_epoch = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=20,
                                     pad_step_number=True)

        with tf.Session(config=config) as sess:
            if tf.train.get_checkpoint_state(FLAGS.train_dir):
                print("Reading model parameters from %s" % FLAGS.train_dir)
                saver.restore(sess, tf.train.latest_checkpoint(FLAGS.train_dir))
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
                    print("global step %d step-time %.2f loss %f ppx_loss %f perplexity %s"
                          % (model.global_step.eval(),
                             (time.time() - start_time) / ((ed - st) / FLAGS.batch_size), loss_step, ppx_step,
                             show(np.exp(ppx_step))))
                    # save the model per training step
                    saver.save(sess, '%s/checkpoint' % FLAGS.train_dir, global_step=model.global_step)
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
                saver_epoch.save(sess, '%s/epoch/checkpoint' % FLAGS.train_dir, global_step=model.global_step)
    else:
        # load test test
        _, _, data_test = load_data(FLAGS.data_dir, is_train=False)
        # load KB & vocab
        raw_vocab, KB_tuple = load_knowledge(FLAGS.data_dir)

        model = Model(word_embed=None, entity_embed=None, vocab_size=FLAGS.vocab_size,
                      num_embed_units=FLAGS.embed_units, num_units=FLAGS.units, num_layers=FLAGS.layers,
                      num_entities=FLAGS.num_entities+FLAGS.num_relations, num_trans_units=FLAGS.trans_units,
                      max_length=FLAGS.max_dec_lens, learning_rate=FLAGS.lr)
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=3,
                               pad_step_number=True)

        with tf.Session(config=config) as sess:
            if FLAGS.inference_version == 0:
                model_path = tf.train.latest_checkpoint(FLAGS.train_dir)
            else:
                model_path = '%s/checkpoint-%08d' % (FLAGS.train_dir, FLAGS.inference_version)
            print("restore from %s" % model_path)
            saver.restore(sess, model_path)

            # test model on test set
            model_test(sess, saver, data_test, KB_tuple)


if __name__ == '__main__':
    main()
