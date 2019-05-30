# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.ops.nn import dynamic_rnn
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell
from tensorflow.contrib.lookup.lookup_ops import MutableHashTable
from tensorflow.python.framework import constant_op
from tensorflow.contrib.layers.python.layers import layers
from .output_projection import output_projection
from .output_projection import loss_computation
from .attention_decoder import attention_decoder_fn_train
from .attention_decoder import attention_decoder_fn_inference
from .attention_decoder import prepare_attention
from .dynamic_decoder import dynamic_rnn_decoder
from .utils import UNK_ID, GO_ID, EOS_ID, NONE_ID


class Model(object):
    def __init__(self, word_embed, entity_embed, vocab_size=30000, num_embed_units=300, num_units=512, num_layers=2,
                 num_entities=0, num_trans_units=100, max_length=60, learning_rate=0.0001, learning_rate_decay_factor=0.95,
                 max_gradient_norm=5.0, num_samples=500, output_alignments=True):
        # initialize params
        self.vocab_size = vocab_size
        self.num_embed_units = num_embed_units
        self.num_units = num_units
        self.num_layers = num_layers
        self.num_entities = num_entities
        self.num_trans_units = num_trans_units
        self.learning_rate = learning_rate
        self.max_gradient_norm = max_gradient_norm
        self.num_samples = num_samples
        self.max_length = max_length
        self.output_alignments = output_alignments

        # build the embedding table (index to vector)
        if word_embed is None:
            # initialize the embedding randomly
            self.word_embed = tf.get_variable('word_embed', [self.vocab_size, self.num_embed_units], tf.float32)
        else:
            # initialize the embedding by pre-trained word vectors
            self.word_embed = tf.get_variable('word_embed', dtype=tf.float32, initializer=word_embed)
        if entity_embed is None:
            # initialize the embedding randomly
            self.entity_trans = tf.get_variable('entity_embed',
                                                [num_entities, num_trans_units], tf.float32, trainable=False)
        else:
            # initialize the embedding by pre-trained trans vectors
            self.entity_trans = tf.get_variable('entity_embed',
                                                dtype=tf.float32, initializer=entity_embed, trainable=False)

        # initialize inputs and outputs
        self.posts = tf.placeholder(tf.string, (None, None), 'enc_inps')  # batch*len
        self.posts_length = tf.placeholder(tf.int32, (None), 'enc_lens')  # batch
        self.responses = tf.placeholder(tf.string, (None, None), 'dec_inps')  # batch*len
        self.responses_length = tf.placeholder(tf.int32, (None), 'dec_lens')  # batch
        self.entities = tf.placeholder(tf.string, (None, None, None), 'entities')  # batch
        self.entity_masks = tf.placeholder(tf.string, (None, None), 'entity_masks')  # batch
        self.triples = tf.placeholder(tf.string, (None, None, None, 3), 'triples')  # batch
        self.posts_triple = tf.placeholder(tf.int32, (None, None, 1), 'enc_triples')  # batch
        self.responses_triple = tf.placeholder(tf.string, (None, None, 3), 'dec_triples')  # batch
        self.match_triples = tf.placeholder(tf.int32, (None, None, None), 'match_triples')  # batch
        self._init_vocabs()

        # build the vocab table (string to index)
        self.posts_word_id = self.symbol2index.lookup(self.posts)   # batch*len
        self.posts_entity_id = self.entity2index.lookup(self.posts)   # batch*len
        self.responses_target = self.symbol2index.lookup(self.responses)  # batch*len
        batch_size, decoder_len = tf.shape(self.responses)[0], tf.shape(self.responses)[1]
        self.responses_word_id = tf.concat([tf.ones([batch_size, 1], dtype=tf.int64)*GO_ID,
                                            tf.split(self.responses_target, [decoder_len-1, 1], 1)[0]], 1) # batch*len
        self.decoder_mask = tf.reshape(tf.cumsum(
            tf.one_hot(self.responses_length-1, decoder_len), reverse=True, axis=1),
            [-1, decoder_len])

        # build entity embeddings
        entity_trans_transformed = tf.layers.dense(self.entity_trans, self.num_trans_units,
                                                        activation=tf.tanh, name='trans_transformation')
        padding_entity = tf.get_variable('entity_padding_embed', [7, self.num_trans_units],
                                         dtype=tf.float32, initializer=tf.zeros_initializer())
        self.entity_embed = tf.concat([padding_entity, entity_trans_transformed], axis=0)

        # get knowledge graph embedding, knowledge triple embedding
        self.triples_embedding, self.entities_word_embedding, self.graph_embedding = self._build_kg_embedding()

        # build knowledge graph
        graph_embed_input, triple_embed_input = self._build_kg_graph()

        # build encoder
        encoder_output, encoder_state = self._build_encoder(graph_embed_input)

        # build decoder
        self._build_decoder(encoder_output, encoder_state, triple_embed_input)

        # initialize training process
        self.global_step = tf.Variable(0, trainable=False)
        self.params = tf.global_variables()

        gradients = tf.gradients(self.decoder_loss, self.params)
        self.clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.update = optimizer.apply_gradients(zip(self.clipped_gradients, self.params), global_step=self.global_step)

        tf.summary.scalar('decoder_loss', self.decoder_loss)
        for each in tf.trainable_variables():
            tf.summary.histogram(each.name, each)
        self.merged_summary_op = tf.summary.merge_all()

    def _init_vocabs(self):
        self.symbol2index = MutableHashTable(key_dtype=tf.string, value_dtype=tf.int64,
                                             default_value=UNK_ID, shared_name="in_table",
                                             name="in_table", checkpoint=True)
        self.index2symbol = MutableHashTable(key_dtype=tf.int64, value_dtype=tf.string,
                                             default_value='_UNK', shared_name="out_table",
                                             name="out_table", checkpoint=True)
        self.entity2index = MutableHashTable(key_dtype=tf.string, value_dtype=tf.int64,
                                             default_value=NONE_ID, shared_name="entity_in_table",
                                             name="entity_in_table", checkpoint=True)
        self.index2entity = MutableHashTable(key_dtype=tf.int64, value_dtype=tf.string,
                                             default_value='_NONE', shared_name="entity_out_table",
                                             name="entity_out_table", checkpoint=True)

    def _build_kg_embedding(self):
        encoder_batch_size, encoder_len = tf.unstack(tf.shape(self.posts))
        triple_num = tf.shape(self.triples)[1]

        triples_embedding = tf.reshape(
            tf.nn.embedding_lookup(self.entity_embed, self.entity2index.lookup(self.triples)),
            [encoder_batch_size, triple_num, -1, 3 * self.num_trans_units])

        entities_word_embedding = tf.reshape(
            tf.nn.embedding_lookup(self.word_embed, self.symbol2index.lookup(self.entities)),
            [encoder_batch_size, -1, self.num_embed_units])

        head, relation, tail = tf.split(triples_embedding, [self.num_trans_units] * 3, axis=3)
        with tf.variable_scope('graph_attention', reuse=tf.AUTO_REUSE):
            head_tail = tf.concat([head, tail], axis=3)
            head_tail_transformed = tf.layers.dense(head_tail, self.num_trans_units,
                                                    activation=tf.tanh, name='head_tail_transform')
            relation_transformed = tf.layers.dense(relation, self.num_trans_units, name='relation_transform')
            e_weight = tf.reduce_sum(relation_transformed * head_tail_transformed, axis=3)
            alpha_weight = tf.nn.softmax(e_weight)
            graph_embedding = tf.reduce_sum(tf.expand_dims(alpha_weight, 3) * head_tail, axis=2)
        return triples_embedding, entities_word_embedding, graph_embedding

    def _build_kg_graph(self):
        encoder_batch_size, encoder_len = tf.unstack(tf.shape(self.posts))
        batch_size, decoder_len = tf.shape(self.responses)[0], tf.shape(self.responses)[1]
        # knowledge graph vectors
        graph_embed_input = tf.gather_nd(
            self.graph_embedding,
            tf.concat(
                [tf.tile(tf.reshape(tf.range(encoder_batch_size, dtype=tf.int32), [-1, 1, 1]),
                         [1, encoder_len, 1]),
                 self.posts_triple], axis=2))

        # knowledge triple vectors
        triple_embed_input = tf.reshape(
            tf.nn.embedding_lookup(self.entity_embed,
                                   self.entity2index.lookup(self.responses_triple)),
            [batch_size, decoder_len, 3 * self.num_trans_units])

        return graph_embed_input, triple_embed_input

    def _build_encoder(self, graph_embed_input):
        post_word_input = tf.nn.embedding_lookup(self.word_embed, self.posts_word_id)  # batch*len*unit
        encoder_cell = MultiRNNCell([GRUCell(self.num_units) for _ in range(self.num_layers)])

        # encoder input: e(x_t) = [w(x_t); g_i]
        encoder_input = tf.concat([post_word_input, graph_embed_input], axis=2)
        encoder_output, encoder_state = dynamic_rnn(encoder_cell, encoder_input,
                                                    self.posts_length, dtype=tf.float32, scope="encoder")
        # shape:[batch_size, max_time, cell.output_size]
        return encoder_output, encoder_state

    def _build_decoder(self, encoder_output, encoder_state, triple_embed_input):
        # decoder input: e(y_t) = [w(y_t); k_j]
        encoder_batch_size, encoder_len = tf.unstack(tf.shape(self.posts))
        response_word_input = tf.nn.embedding_lookup(self.word_embed, self.responses_word_id)  # batch*len*unit
        decoder_input = tf.concat([response_word_input, triple_embed_input], axis=2)
        print("decoder_input:", decoder_input.shape)

        # define cell
        decoder_cell = MultiRNNCell([GRUCell(self.num_units) for _ in range(self.num_layers)])

        # get loss functions
        sequence_loss, total_loss = loss_computation(self.vocab_size,
                                                     num_samples=self.num_samples)

        # decoder training process
        with tf.variable_scope('decoder'):
            # prepare attention
            attention_keys, attention_values, attention_score_fn, attention_construct_fn \
                = prepare_attention(encoder_output, 'bahdanau', self.num_units, scope_name="decoder",
                                    imem=(self.graph_embedding, self.triples_embedding),
                                    output_alignments=self.output_alignments)
            print("graph_embedding:", self.graph_embedding.shape)
            print("triples_embedding:", self.triples_embedding.shape)
            decoder_fn_train = attention_decoder_fn_train(encoder_state,
                                                           attention_keys,
                                                           attention_values,
                                                           attention_score_fn,
                                                           attention_construct_fn,
                                                           output_alignments=self.output_alignments,
                                                           max_length=tf.reduce_max(self.responses_length))
            # train decoder
            decoder_output, _, decoder_context_state = dynamic_rnn_decoder(decoder_cell,
                                                                            decoder_fn_train,
                                                                            decoder_input,
                                                                            self.responses_length,
                                                                            scope="decoder_rnn")
            output_fn, selector_fn = output_projection(self.vocab_size,
                                                         scope_name="decoder_rnn")
            output_logits = output_fn(decoder_output)
            selector_logits = selector_fn(decoder_output)
            print("decoder_output:", decoder_output.shape)  # shape: [batch, seq, num_units]
            print("output_logits:", output_logits.shape)
            print("selector_fn:", selector_logits.name)

            triple_len = tf.shape(self.triples)[2]
            one_hot_triples = tf.one_hot(self.match_triples, triple_len)
            use_triples = tf.reduce_sum(one_hot_triples, axis=[2, 3])
            alignments = tf.transpose(decoder_context_state1.stack(), perm=[1, 0, 2, 3])
            self.decoder_loss, self.ppx_loss, self.sentence_ppx \
                = total_loss(output_logits,
                             selector_logits,
                             self.responses_target,
                             self.decoder_mask,
                             alignments,
                             use_triples,
                             one_hot_triples)
            self.sentence_ppx = tf.identity(self.sentence_ppx, name="ppx_loss")

        # decoder inference process
        with tf.variable_scope('decoder', reuse=True):
            # prepare attention
            attention_keys, attention_values, attention_score_fn, attention_construct_fn \
                = prepare_attention(encoder_output, 'bahdanau', self.num_units, scope_name="decoder",
                                    imem=(self.graph_embedding, self.triples_embedding),
                                    output_alignments=self.output_alignments,
                                    reuse=True)
            output_fn, selector_fn = output_projection(self.vocab_size,
                                                         scope_name=None,
                                                         reuse=True)
            decoder_fn_inference \
                = attention_decoder_fn_inference(output_fn, encoder_state,
                                                 attention_keys, attention_values,
                                                 attention_score_fn, attention_construct_fn,
                                                 self.word_embed, GO_ID, EOS_ID, self.max_length, self.vocab_size,
                                                 imem=(self.entities_word_embedding,
                                                       tf.reshape(self.triples_embedding,
                                                                  [encoder_batch_size, -1, 3 * self.num_trans_units])),
                                                 selector_fn=selector_fn)

            # get decoder output
            decoder_distribution, _, infer_context_state \
                = dynamic_rnn_decoder(decoder_cell, decoder_fn_inference, scope="decoder_rnn")
            
            output_len = tf.shape(decoder_distribution)[1]
            output_ids = tf.transpose(infer_context_state.gather(tf.range(output_len)))
            word_ids = tf.cast(tf.clip_by_value(output_ids, 0, self.vocab_size), tf.int64)
            entity_ids = tf.reshape(
                tf.clip_by_value(-output_ids, 0, self.vocab_size) +
                tf.reshape(tf.range(encoder_batch_size) * tf.shape(self.entities_word_embedding)[1], [-1, 1]),
                [-1])
            entities = tf.reshape(
                tf.gather(tf.reshape(self.entities, [-1]), entity_ids),
                [-1, output_len])
            words = self.index2symbol.lookup(word_ids)
            self.generation = tf.where(output_ids > 0, words, entities)
            self.generation = tf.identity(self.generation, name='generation')

    def set_vocabs(self, session, vocab, entity_vocab, relation_vocab):
        op_in = self.symbol2index.insert(constant_op.constant(vocab),
                                         constant_op.constant(list(range(self.vocab_size)), dtype=tf.int64))
        session.run(op_in)
        op_out = self.index2symbol.insert(constant_op.constant(list(range(self.vocab_size)), dtype=tf.int64),
                                          constant_op.constant(vocab))
        session.run(op_out)
        op_in = self.entity2index.insert(constant_op.constant(entity_vocab + relation_vocab),
                                         constant_op.constant(list(range(len(entity_vocab) + len(relation_vocab))),
                                                               dtype=tf.int64))
        session.run(op_in)
        op_out = self.index2entity.insert(constant_op.constant(list(range(len(entity_vocab) + len(relation_vocab))),
                                                                dtype=tf.int64),
                                          constant_op.constant(entity_vocab + relation_vocab))
        session.run(op_out)
        return session

    def print_parameters(self):
        for item in self.params:
            print('%s: %s' % (item.name, item.get_shape().as_list()))
    
    def step_train(self, session, data, forward_only=False, summary=False):
        input_feed = {self.posts: data['posts'],
                      self.posts_length: data['posts_length'],
                      self.responses: data['responses'],
                      self.responses_length: data['responses_length'],
                      self.triples: data['triples'],
                      self.posts_triple: data['posts_triple'],
                      self.responses_triple: data['responses_triple'],
                      self.match_triples: data['match_triples']}
        if forward_only:
            output_feed = [self.sentence_ppx]
        else:
            output_feed = [self.sentence_ppx, self.decoder_loss, self.update]
        if summary:
            output_feed.append(self.merged_summary_op)

        return session.run(output_feed, input_feed)
