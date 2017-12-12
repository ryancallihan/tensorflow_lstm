from enum import Enum
import tensorflow as tf
from tensorflow.contrib import rnn


class Phase(Enum):
    Train = 0
    Validation = 1
    Predict = 2


def lstm_layers(
        config,
        inputs,
        phase=Phase.Predict):
    cells = [tf.nn.rnn_cell.LSTMCell(state_size, activation=tf.nn.tanh, state_is_tuple=True) for state_size in config.hidden_sizes]
    if phase == Phase.Train:
        cells = [rnn.DropoutWrapper(cell, output_keep_prob=config.lstm_output_dropout,
                                    state_keep_prob=config.lstm_state_dropout) for cell in cells]
    cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
    init_state = cell.zero_state(config.batch_size, tf.float32)
    return tf.nn.dynamic_rnn(cell, inputs, initial_state=init_state)


class Model:
    def __init__(
            self,
            config,
            batch,
            vocab_size,
            phase=Phase.Predict):
        batch_size = batch.shape[1]
        input_size = batch.shape[2]

        regularizer = tf.contrib.layers.l2_regularizer(scale=0.11)

        self._x = tf.placeholder(tf.int32, shape=[batch_size, input_size])

        self._lens = tf.placeholder(tf.int32, shape=[batch_size])

        if phase != Phase.Predict:
            self._y = tf.placeholder(
                tf.float32, shape=[batch_size, config.label_size])

        embeds = tf.get_variable("embeddings", shape=[vocab_size, config.char_embed_size])
        input_layer = tf.nn.embedding_lookup(embeds, self._x, None)

        if phase == Phase.Train:
            input_layer = tf.nn.dropout(input_layer, keep_prob=config.input_dropout)

        hidden_layers, final_state = lstm_layers(config, input_layer, phase=phase)

        if phase == Phase.Train:
            hidden_layers = tf.contrib.layers.batch_norm(hidden_layers, center=True, scale=True, is_training=True,
                                                         scope='bn')

        W = tf.get_variable("w", regularizer=regularizer, shape=[config.hidden_sizes[-1], config.label_size])
        b = tf.get_variable("b", shape=[config.label_size])

        hidden_layers = tf.transpose(hidden_layers, [1, 0, 2])
        hidden_layers = tf.gather(hidden_layers, int(hidden_layers.get_shape()[0]) - 1)

        # hidden_layers = tf.layers.dense(hidden_layers, config.hidden_sizes[2])

        self._logits = logits = tf.matmul(hidden_layers, W) + b

        if phase != Phase.Predict:
            losses = tf.nn.softmax_cross_entropy_with_logits(
                labels=self._y, logits=logits)
            self._loss = tf.reduce_sum(losses)

        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)

        if phase == Phase.Train:
            reg_term += tf.nn.l2_loss(embeds) * 0.15
            self._train_op = tf.train.AdamOptimizer(learning_rate=config.start_lr) \
                .minimize(self._loss + reg_term)
            self._probs = tf.nn.softmax(logits)

        self._predicted_labels = predicted_labels = tf.argmax(logits, axis=1)

        if phase != Phase.Predict:
            self._hp_labels = hp_labels = tf.argmax(self.y, axis=1)
            correct = tf.equal(hp_labels, predicted_labels)
            correct = tf.cast(correct, tf.float32)
            self._accuracy = tf.reduce_mean(correct)

    @property
    def logits(self):
        return self._logits

    @property
    def predicted_labels(self):
        return self._predicted_labels

    @property
    def hp_labels(self):
        return self._hp_labels

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def lens(self):
        return self._lens

    @property
    def loss(self):
        return self._loss

    @property
    def probs(self):
        return self._probs

    @property
    def train_op(self):
        return self._train_op

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y
