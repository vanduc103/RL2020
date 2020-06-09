import tensorflow as tf
import numpy as np

def build_network(maxlen=5, input_dim=29, vocab_size=36, hidden_dim=64, embed_dim=64):
    states = tf.placeholder(tf.float32, [None, input_dim], name="states")
    actions = tf.placeholder(tf.int64, [None, maxlen], name="actions")
    batch_size = tf.shape(states)[0] # get batch size as a tensor (N)

    # actions input = actions except the last word (<END>)
    X = actions[:, :-1]

    # compute the initial state from data states => use an NN layer
    h0 = tf.layers.dense(states, hidden_dim) # [N, hidden_dim]

    # compute word embedding from words in actions to embedding vector => [N, T, embed_dim]
    embed = tf.contrib.layers.embed_sequence(ids=X, vocab_size=36, embed_dim=embed_dim)

    # use RNN to compute output from initial state h0 => outputs [N, T, hidden_dim]
    rnn_cell = tf.contrib.rnn.BasicRNNCell(hidden_dim)
    rnn_outputs, rnn_states = tf.nn.dynamic_rnn(rnn_cell, embed, initial_state=h0, dtype=tf.float32)

    # convert rnn output to vocab size
    rnn_outputs = tf.reshape(rnn_outputs, [batch_size * (maxlen-1), hidden_dim]) # [N, T, H] => [N*T, H]
    probs = tf.nn.softmax(tf.layers.dense(rnn_outputs, vocab_size)) # [N*T, Vocab_Size]
    outputs = tf.reshape(probs, [batch_size, maxlen-1, vocab_size], name="outputs") # [N, T, Vocab_Size]

    return states, actions, outputs
