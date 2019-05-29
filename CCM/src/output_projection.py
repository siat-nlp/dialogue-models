import tensorflow as tf
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.ops import variable_scope


def output_projection(num_symbols, scope_name=None, reuse=False):
    """
       Define functions to compute output.
       """
    def output_fn(outputs):
        if scope_name is not None:
            with variable_scope.variable_scope("%s/output_projection" % scope_name, reuse=reuse) as scope:
                output_logits = layers.linear(outputs, num_symbols, scope=scope)
        else:
            with variable_scope.variable_scope("output_projection", reuse=reuse) as scope:
                output_logits = layers.linear(outputs, num_symbols, scope=scope)
        return output_logits

    def selector_fn(outputs):
        if scope_name is not None:
            with variable_scope.variable_scope("%s/selector" % scope_name, reuse=reuse) as scope:
                selector_logits = layers.linear(outputs, 1, scope=scope)
        else:
            with variable_scope.variable_scope("selector", reuse=reuse) as scope:
                selector_logits = layers.linear(outputs, 1, scope=scope)
        return selector_logits

    return output_fn, selector_fn


def loss_computation(num_symbols, num_samples=None, scope=None):
    """
    Define functions to compute loss.
    """
    def sequence_loss(output_logits, targets, masks):
        logits = tf.reshape(output_logits, [-1, num_symbols])
        local_labels = tf.reshape(targets, [-1])
        local_masks = tf.reshape(masks, [-1])

        local_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=local_labels, logits=logits)
        local_loss = local_loss * local_masks

        loss = tf.reduce_sum(local_loss)
        total_size = tf.reduce_sum(local_masks)
        total_size += 1e-12  # to avoid division by 0 for all-0 weights

        return loss / total_size

    def total_loss(output_logits, selector_logits, targets, masks, alignments, use_entities, entity_targets):
        batch_size = tf.shape(output_logits)[0]
        local_masks = tf.reshape(masks, [-1])
        one_hot_targets = tf.one_hot(targets, num_symbols)

        word_prob = tf.reduce_sum(tf.nn.softmax(output_logits) * one_hot_targets, axis=2)
        selector = tf.squeeze(tf.sigmoid(selector_logits))
        triple_prob = tf.reduce_sum(alignments * entity_targets, axis=[2, 3])
        ppx_prob = word_prob * (1 - use_entities) + triple_prob * use_entities
        final_prob = word_prob * (1 - selector) * (1 - use_entities) + triple_prob * selector * use_entities

        final_loss = tf.reduce_sum(tf.reshape(-tf.log(1e-12 + final_prob), [-1]) * local_masks)
        ppx_loss = tf.reduce_sum(tf.reshape(-tf.log(1e-12 + ppx_prob), [-1]) * local_masks)
        sentence_ppx = tf.reduce_sum(
            tf.reshape(tf.reshape(-tf.log(1e-12 + ppx_prob), [-1]) * local_masks, [batch_size, -1]), axis=1)
        selector_loss = tf.reduce_sum(
            tf.reshape(-tf.log(1e-12 + selector * use_entities + (1 - selector) * (1 - use_entities)),
                       [-1]) * local_masks)

        loss = final_loss + selector_loss
        total_size = tf.reduce_sum(local_masks)
        total_size += 1e-12  # to avoid division by 0 for all-0 weights

        decoder_loss = loss / total_size
        ppx_loss = ppx_loss / total_size
        sentence_ppx = sentence_ppx / tf.reduce_sum(masks, axis=1)

        return decoder_loss, ppx_loss, sentence_ppx

    return sequence_loss, total_loss
