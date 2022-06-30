import sys
sys.path.append('../../../src')

import os.path
import numpy as np
import tensorflow as tf

from batcher import Batcher

batch_size = 64
cost_batch_size = 10_000
n_iterations = 1_000_000
display_step = 10_000
lstm_size = 128
n_layers = 3


def create_nn(bin_dir, out_dir, file_prefix):
    """
    :param bin_dir folder where binaries are stored
    :param out_dir folder where store the result of the nn training
    :param file_prefix prefix of nn files (lefty, righty, decl, dummy)
    """

    model_path = os.path.join(out_dir, file_prefix)

    x_train = np.load(os.path.join(bin_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(bin_dir, 'Y_train.npy'))

    x_val = np.load(os.path.join(bin_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(bin_dir, 'Y_val.npy'))

    n_examples = y_train.shape[0]
    n_ftrs = x_train.shape[2]
    n_cards = y_train.shape[2]

    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    cells = []
    for _ in range(n_layers):
        cell = tf.contrib.rnn.DropoutWrapper(
            tf.nn.rnn_cell.BasicLSTMCell(lstm_size),
            output_keep_prob=keep_prob
        )
        cells.append(cell)

    state = []
    for i, cell_i in enumerate(cells):
        s_c = tf.placeholder(tf.float32, [1, lstm_size], name='state_c_{}'.format(i))
        s_h = tf.placeholder(tf.float32, [1, lstm_size], name='state_h_{}'.format(i))
        state.append(tf.contrib.rnn.LSTMStateTuple(c=s_c, h=s_h))
    state = tuple(state)

    x_in = tf.placeholder(tf.float32, [1, n_ftrs], name='x_in')

    lstm_cell = tf.contrib.rnn.MultiRNNCell(cells)

    seq_in = tf.placeholder(tf.float32, [None, None, n_ftrs], 'seq_in')
    seq_out = tf.placeholder(tf.float32, [None, None, n_cards], 'seq_out')

    softmax_w = tf.get_variable('softmax_w', shape=[lstm_cell.output_size, n_cards], dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer(seed=1337))

    out_rnn, _ = tf.nn.dynamic_rnn(lstm_cell, seq_in, dtype=tf.float32)

    out_card_logit = tf.matmul(tf.reshape(out_rnn, [-1, lstm_size]), softmax_w, name='out_card_logit')
    out_card_target = tf.reshape(seq_out, [-1, n_cards], name='out_card_target')

    output, next_state = lstm_cell(x_in, state)

    out_card = tf.nn.softmax(tf.matmul(output, softmax_w), name='out_card')

    for i, next_i in enumerate(next_state):
        tf.identity(next_i.c, name='next_c_{}'.format(i))
        tf.identity(next_i.h, name='next_h_{}'.format(i))

    cost = tf.losses.softmax_cross_entropy(out_card_target, out_card_logit)

    train_step = tf.train.AdamOptimizer(0.0003).minimize(cost)

    batch = Batcher(n_examples, batch_size)
    cost_batch = Batcher(n_examples, cost_batch_size)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(max_to_keep=50)

        for i in range(n_iterations):
            x_batch, y_batch = batch.next_batch([x_train, y_train])
            if i % display_step == 0:
                x_cost, y_cost = cost_batch.next_batch([x_train, y_train])
                c_train = sess.run(cost, feed_dict={seq_in: x_cost, seq_out: y_cost, keep_prob: 1.0})
                c_valid = sess.run(cost, feed_dict={seq_in: x_val, seq_out: y_val, keep_prob: 1.0})

                print('{}. c_train={} c_valid={}'.format('{}/{}'.format(i, n_iterations), c_train, c_valid))

                sys.stdout.flush()
                saver.save(sess, model_path, global_step=i)

            sess.run(train_step, feed_dict={seq_in: x_batch, seq_out: y_batch, keep_prob: 0.8})

        saver.save(sess, model_path, global_step=n_iterations)


def continue_nn(bin_dir, checkpoint_model, out_dir, file_prefix):
    """
    :param bin_dir folder where binaries are stored
    :param checkpoint_model
    :param out_dir folder where store the result of the nn training
    :param file_prefix prefix of nn files (lefty, righty, decl, dummy)
    """

    model_path = os.path.join(out_dir, file_prefix)
    start_iteration = int(checkpoint_model.split('-')[-1])

    x_train = np.load(os.path.join(bin_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(bin_dir, 'Y_train.npy'))

    x_val = np.load(os.path.join(bin_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(bin_dir, 'Y_val.npy'))

    n_examples = y_train.shape[0]
    n_ftrs = x_train.shape[2]
    n_cards = y_train.shape[2]

    graph = tf.Graph()
    sess = tf.Session(graph=graph)

    with graph.as_default():
        saver = tf.train.import_meta_graph(f'{checkpoint_model}.meta')
        saver.restore(sess, checkpoint_model)

        seq_in = graph.get_tensor_by_name('seq_in:0')
        seq_out = graph.get_tensor_by_name('seq_out:0')

        out_card_logit = graph.get_tensor_by_name('out_card_logit:0')
        out_card_target = graph.get_tensor_by_name('out_card_target:0')

        keep_prob = graph.get_tensor_by_name('keep_prob:0')

        cost = tf.losses.softmax_cross_entropy(out_card_target, out_card_logit)

        train_step = graph.get_operation_by_name('Adam')

        batch = Batcher(n_examples, batch_size)
        cost_batch = Batcher(n_examples, cost_batch_size)

        saver = tf.train.Saver()

        for i in range(start_iteration, start_iteration + n_iterations):
            x_batch, y_batch = batch.next_batch([x_train, y_train])
            if i % display_step == 0:
                x_cost, y_cost = cost_batch.next_batch([x_train, y_train])
                c_train = sess.run(cost, feed_dict={seq_in: x_cost, seq_out: y_cost, keep_prob: 1.0})
                c_valid = sess.run(cost, feed_dict={seq_in: x_val, seq_out: y_val, keep_prob: 1.0})

                print('{}. c_train={} c_valid={}'.format('{}/{}'.format(i, start_iteration + n_iterations), c_train, c_valid))

                sys.stdout.flush()
                saver.save(sess, model_path, global_step=i)

            sess.run(train_step, feed_dict={seq_in: x_batch, seq_out: y_batch, keep_prob: 0.8})

        saver.save(sess, model_path, global_step=n_iterations)