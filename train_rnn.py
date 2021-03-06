#!/usr/bin/env python
import os

import tensorflow as tf
import sys
import random
import numpy as np
import time
from env import Env
from model_rnn import build_network

flags = tf.app.flags

flags.DEFINE_string('experiment', 'rnn_lr0.1', 'Name of the current experiment')
flags.DEFINE_string('dataset_path', '', 'Path to dataset to run')
flags.DEFINE_integer('tmax', 100, 'Number of training timesteps.')
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_integer('batch_size', 1, 'Batch size')
flags.DEFINE_integer('anneal_epsilon_timesteps', 10, 'Number of timesteps to anneal epsilon.')
flags.DEFINE_string('summary_dir', './output/summaries', 'Directory for storing tensorboard summaries')
flags.DEFINE_string('checkpoint_dir', './output/checkpoints', 'Directory for storing model checkpoints')
flags.DEFINE_integer('summary_interval', 20,
                     'Save training summary to file every n seconds (rounded '
                     'up to statistics interval.')
flags.DEFINE_integer('checkpoint_interval', 10,
                     'Checkpoint the model (i.e. save the parameters) every n '
                     'seconds (rounded up to statistics interval.')
flags.DEFINE_boolean('testing', False, 'If true, run model evaluation')
flags.DEFINE_string('checkpoint_path', './output/checkpoints/rnn_lr0.1.ckpt-100', 'Path to recent checkpoint to use for evaluation')
flags.DEFINE_string('eval_dir', './output/', 'Directory to store gym evaluation')
flags.DEFINE_integer('num_eval_episodes', 1, 'Number of episodes to run env evaluation.')
FLAGS = flags.FLAGS
TMAX = FLAGS.tmax

def sample_final_epsilon():
    """
    Sample a final epsilon value to anneal towards from a distribution.
    """
    final_epsilons = np.array([.1,.01,.5])
    probabilities = np.array([0.4,0.3,0.3])
    return np.random.choice(final_epsilons, 1, p=list(probabilities))[0]

def choose_next_action(epsilon, readout_t):
    # Choose next action based on e-greedy policy
    a_t = np.zeros_like(readout_t)
    action_index = np.zeros(readout_t.shape[0]).astype(int)
    num_actions = readout_t.shape[1]
    for i in range(readout_t.shape[0]):
        action_index[i] = 0
        if random.random() <= epsilon:
            action_index[i] = random.randrange(num_actions)
        else:
            action_index[i] = np.argmax(readout_t[i])
        a_t[i, action_index[i]] = 1

    return a_t, action_index

def run(env, session, graph_ops, num_actions, summary_ops, saver):
    """
    Policy Gradient running
    """

    # Unpack graph ops
    s = graph_ops["s"]
    action_input = graph_ops["action_input"]
    policy_values = graph_ops["policy_values"]
    a = graph_ops["a"]
    a_rescaling, a_preprocessor, a_classifier = a[0], a[1], a[2]
    y = graph_ops["y"]
    grad_update = graph_ops["grad_update"]
    cost = graph_ops["cost"]
    # the length of action sequence
    maxlen = 5

    summary_placeholders, update_ops, summary_op = summary_ops


    # Initialize network gradients
    s_batch = []
    action_batch = []
    rescaling_batch = []
    preprocessor_batch = []
    classifier_batch = []
    y_batch = []

    final_epsilon = sample_final_epsilon()
    initial_epsilon = 1.0
    epsilon = 1.0

    print("Starting with final epsilon", final_epsilon)

    T = 0
    t = 0
    batch_size = FLAGS.batch_size
    min_loss = 9999
    while T < TMAX:
        T += 1
        for b in range(int(env.size/batch_size)):
            # Get data state
            s_t = env.data_state(b, batch_size) 
            terminal = False

            # Set up per-episode counters
            ep_reward = 0
            episode_ave_max_q = 0
            ep_t = 0

            while True:
                # Forward the deep policy network, get policy values
                policy_rescaling, policy_preprocessor, policy_classifier = policy_values[0], policy_values[1], policy_values[2]
                pr_actions = np.zeros((np.array(s_t).shape[0],maxlen), dtype=np.int64)
                pr_actions[:, 0] = env.action_vocab['<START>']
                outrescaling_t = policy_rescaling.eval(session = session, feed_dict = {s : s_t, action_input: pr_actions})
                pr_actions[:, 1] = np.argmax(outrescaling_t)
                outpreprocessor_t = policy_preprocessor.eval(session = session, feed_dict = {s : s_t, action_input: pr_actions})
                pr_actions[:, 2] = np.argmax(outpreprocessor_t)
                outclassifier_t = policy_classifier.eval(session = session, feed_dict = {s : s_t, action_input: pr_actions})
                action_batch.append(pr_actions)
                action_batch = np.reshape(np.array(action_batch), [-1, np.array(pr_actions).shape[1]])

                # Choose next action based on e-greedy policy
                rescaling_t, rescaling_action = choose_next_action(epsilon, outrescaling_t)
                preprocessor_t, preprocessor_action = choose_next_action(epsilon, outpreprocessor_t)
                classifier_t, classifier_action = choose_next_action(epsilon, outclassifier_t)
                
                # Scale down epsilon
                if epsilon > final_epsilon:
                    epsilon -= (initial_epsilon - final_epsilon) / FLAGS.anneal_epsilon_timesteps
        
                # Action execution
                s_t1, r_t, terminal, info = env.run_actions((rescaling_action, preprocessor_action, classifier_action), b, batch_size)

                # Batch update
                y_batch.append(-r_t)
                y_batch = np.reshape(np.array(y_batch), [-1])
        
                rescaling_batch.append(rescaling_t)
                rescaling_batch = np.reshape(np.array(rescaling_batch), [-1, np.array(rescaling_t).shape[1]])
                preprocessor_batch.append(preprocessor_t)
                preprocessor_batch = np.reshape(np.array(preprocessor_batch), [-1, np.array(preprocessor_t).shape[1]])
                classifier_batch.append(classifier_t)
                classifier_batch = np.reshape(np.array(classifier_batch), [-1, np.array(classifier_t).shape[1]])

                s_batch.append(s_t)
                s_batch = np.reshape(np.array(s_batch), [-1, np.array(s_t).shape[1]])
        
                # Update the state and counters
                s_t = s_t1
                t += 1

                ep_t += 1
                ep_reward += r_t
                episode_ave_max_q += np.max(outrescaling_t) + np.max(outpreprocessor_t) + np.max(outclassifier_t)

                print("T =",T,", t =",t, ", b =",b)
                loss = 0
                if len(s_batch) > 0:
                    print("===========run(grad_update)===========")
                    _, loss = session.run([grad_update, cost], feed_dict = {y : y_batch,
                                                                          a_rescaling : rescaling_batch,
                                                                          a_preprocessor: preprocessor_batch,
                                                                          a_classifier: classifier_batch,
                                                                          action_input: action_batch,
                                                                          s : s_batch})
                # Clear gradients
                s_batch = []
                action_batch = []
                rescaling_batch = []
                preprocessor_batch = []
                classifier_batch = []
                y_batch = []
        
                # Save model progress
                if loss < min_loss:
                    print("==========save(session), reward=", r_t, "loss=", loss)
                    min_loss = loss
                    saver.save(session, FLAGS.checkpoint_dir+"/"+FLAGS.experiment+".ckpt", global_step = t)

                # End an episode
                break

# Build model graph
def build_graph(state_dim, vocab_size, num_actions):
    # Create policy network
    s, action_input, policy = build_network(maxlen=5, vocab_size=vocab_size, input_dim=state_dim, hidden_dim=64, embed_dim=64)
    rescaling_num, preprocessor_num, classifier_num = num_actions[0], num_actions[1], num_actions[2]
    policy_rescaling = tf.nn.softmax(tf.layers.dense(policy[:,0,:], rescaling_num, activation="relu"))
    policy_preprocessor = tf.nn.softmax(tf.layers.dense(policy[:,1,:], preprocessor_num, activation="relu"))
    policy_classifier = tf.nn.softmax(tf.layers.dense(policy[:,2,:], classifier_num, activation="relu"))
    policy_values = [policy_rescaling, policy_preprocessor, policy_classifier]
    network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    # Define cost and gradient update op
    a_rescaling = tf.placeholder("float", [None, rescaling_num]) # mask vector for action
    a_preprocessor = tf.placeholder("float", [None, preprocessor_num]) # mask vector for action
    a_classifier = tf.placeholder("float", [None, classifier_num]) # mask vector for action
    a = [a_rescaling, a_preprocessor, a_classifier]
    y = tf.placeholder("float", [None]) # discounted reward
    action_pred_values = 0
    for i in range(3):
        action_pred_values += tf.reduce_sum(tf.multiply(policy_values[i], a[i]), reduction_indices=1)

    l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in tf.trainable_variables()  ])
    pg_loss = tf.reduce_mean(-tf.log(action_pred_values)*y)

    cost = pg_loss + 0.002 * l2_loss
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grad_update = optimizer.minimize(cost, var_list=network_params)

    graph_ops = {"s" : s,
                 "action_input" : action_input,
                 "policy_values" : policy_values,
                 "a" : a,
                 "y" : y,
                 "grad_update" : grad_update,
                 "cost": cost}

    return graph_ops

# Set up some episode summary ops to visualize on tensorboard.
def setup_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Episode Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Max Q Value", episode_ave_max_q)
    logged_epsilon = tf.Variable(0.)
    tf.summary.scalar("Epsilon", logged_epsilon)
    logged_T = tf.Variable(0.)
    summary_vars = [episode_reward, episode_ave_max_q, logged_epsilon]
    summary_placeholders = [tf.placeholder("float") for i in range(len(summary_vars))]
    update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
    summary_op = tf.summary.merge_all()
    return summary_placeholders, update_ops, summary_op


def train(session, graph_ops, num_actions, saver):
    # Set up environment
    env = Env(dataset_path=FLAGS.dataset_path)
    
    summary_ops = setup_summaries()
    summary_op = summary_ops[-1]

    # Initialize variables
    session.run(tf.initialize_all_variables())
    summary_save_path = FLAGS.summary_dir + "/" + FLAGS.experiment
    writer = tf.summary.FileWriter(summary_save_path, session.graph)
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    # Run the Policy Gradient
    run(env, session, graph_ops, num_actions, summary_ops, saver)

def evaluation(session, graph_ops, saver):
    saver.restore(session, FLAGS.checkpoint_path)
    print("Restored model weights from ", FLAGS.checkpoint_path)
    env = Env(dataset_path=FLAGS.dataset_path)

    # Unpack graph ops
    s = graph_ops["s"]
    action_input = graph_ops["action_input"]
    policy_values = graph_ops["policy_values"]

    for i_episode in range(FLAGS.num_eval_episodes):
        ep_reward = []
        batch_size = 1
        for b in range(int(env.size/batch_size)):
            s_t = env.data_state(b, batch_size)
            maxlen = 5
            policy_rescaling, policy_preprocessor, policy_classifier = policy_values[0], policy_values[1], policy_values[2]
            pr_actions = np.zeros((np.array(s_t).shape[0],maxlen),dtype=np.int64)
            pr_actions[:, 0] = env.action_vocab['<START>']
            outrescaling_t = policy_rescaling.eval(session = session, feed_dict = {s : s_t, action_input: pr_actions})
            pr_actions[:, 1] = np.argmax(outrescaling_t)
            outpreprocessor_t = policy_preprocessor.eval(session = session, feed_dict = {s : s_t, action_input: pr_actions})
            pr_actions[:, 2] = np.argmax(outpreprocessor_t)
            outclassifier_t = policy_classifier.eval(session = session, feed_dict = {s : s_t, action_input: pr_actions})

            epsilon = 0.0
            rescaling_t, rescaling_action = choose_next_action(epsilon, outrescaling_t)
            preprocessor_t, preprocessor_action = choose_next_action(epsilon, outpreprocessor_t)
            classifier_t, classifier_action = choose_next_action(epsilon, outclassifier_t)

            s_t1, r_t, terminal, info = env.run_actions((rescaling_action, preprocessor_action, classifier_action), b, batch_size)
            s_t = s_t1
            ep_reward.append(r_t)
        print("Average reward ", sum(ep_reward)/len(ep_reward))

def main():
    tf.reset_default_graph()
    session = tf.InteractiveSession()
    session.run(tf.global_variables_initializer())

    vocab_size = 36
    state_dim, rescaling_num, preprocessor_num, classifier_num = 29, 6, 13, 15
    num_actions = [rescaling_num, preprocessor_num, classifier_num]
    graph_ops = build_graph(state_dim, vocab_size, num_actions)
    saver = tf.train.Saver()

    if FLAGS.testing:
        evaluation(session, graph_ops, saver)
    else:
        train(session, graph_ops, num_actions, saver)


if __name__ == "__main__":
    main()
