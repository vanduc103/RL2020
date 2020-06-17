#!/usr/bin/env python
import os

import tensorflow as tf
import sys
import random
import numpy as np
import time
from env import Env
from model import build_network

flags = tf.app.flags

flags.DEFINE_string('experiment', 'fc_lr0.1', 'Name of the current experiment')
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
flags.DEFINE_boolean('testing', False, 'If true, run evaluation')
flags.DEFINE_string('checkpoint_path', './output/checkpoints/fc_lr0.1.ckpt-100', 'Path to recent checkpoint to use for evaluation')
flags.DEFINE_string('eval_dir', './output/', 'Directory to store model evaluation')
flags.DEFINE_integer('num_eval_episodes', 1, 'Number of episodes to run model evaluation.')
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
    policy_values = graph_ops["policy_values"]
    a = graph_ops["a"]
    a_rescaling, a_preprocessor, a_classifier = a[0], a[1], a[2]
    y = graph_ops["y"]
    grad_update = graph_ops["grad_update"]
    cost = graph_ops["cost"]

    summary_placeholders, update_ops, summary_op = summary_ops

    # Initialize network gradients
    s_batch = []
    rescaling_batch = []
    preprocessor_batch = []
    classifier_batch = []
    y_batch = []

    final_epsilon = sample_final_epsilon()
    initial_epsilon = 1.0
    epsilon = 1.0

    print("Starting running with final epsilon", final_epsilon)

    T = 0
    t = 0
    batch_size = FLAGS.batch_size
    min_loss = 9999
    while T < TMAX:
        T += 1
        for b in range(int(env.size/batch_size)):
            # Get initial state
            s_t = env.data_state(b, batch_size)
            terminal = False

            # Set up per-episode counters
            ep_reward = 0
            episode_ave_max_q = 0
            ep_t = 0

            while True:
                # Forward the deep policy network, get policy values
                policy_rescaling, policy_preprocessor, policy_classifier = policy_values[0], policy_values[1], policy_values[2]
                outrescaling_t = policy_rescaling.eval(session = session, feed_dict = {s : s_t})
                outpreprocessor_t = policy_preprocessor.eval(session = session, feed_dict = {s : s_t})
                outclassifier_t = policy_classifier.eval(session = session, feed_dict = {s : s_t})
                
                # Choose next action based on e-greedy policy
                rescaling_num, preprocessor_num, classifier_num = num_actions[0], num_actions[1], num_actions[2]
                rescaling_t, rescaling_action = choose_next_action(epsilon, outrescaling_t)
                preprocessor_t, preprocessor_action = choose_next_action(epsilon, outpreprocessor_t)
                classifier_t, classifier_action = choose_next_action(epsilon, outclassifier_t)

                # Scale down epsilon
                if epsilon > final_epsilon:
                    epsilon -= (initial_epsilon - final_epsilon) / FLAGS.anneal_epsilon_timesteps
        
                # Action execution
                s_t1, r_t, terminal, info = env.run_actions((rescaling_action, preprocessor_action, classifier_action), b, batch_size)

                y_batch.append(-r_t)
                y_batch = np.reshape(np.array(y_batch), [-1])

                # Update batch
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
                    print("===========run(grad_update)==============")
                    _, loss = session.run([grad_update, cost], feed_dict = {y : y_batch,
                                                          a_rescaling : rescaling_batch,
                                                          a_preprocessor: preprocessor_batch,
                                                          a_classifier: classifier_batch,
                                                          s : s_batch})
                # Clear gradients
                s_batch = []
                rescaling_batch = []
                preprocessor_batch = []
                classifier_batch = []
                y_batch = []

                # Save model progress
                if loss < min_loss:
                    print("==========save(session), reward=", r_t, "loss=", loss)
                    saver.save(session, FLAGS.checkpoint_dir+"/"+FLAGS.experiment+".ckpt", global_step = t)

                # End an episode
                break

# Build model graph
def build_graph(input_shape, num_actions):
    rescaling_num, preprocessor_num, classifier_num = num_actions[0], num_actions[1], num_actions[2]
    # Create policy network
    s, policy_rescaling, policy_preprocessor, policy_classifier = build_network(input_shape, rescaling_num, preprocessor_num, classifier_num, name="policy_network")
    network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy_network")
    policy_values = [policy_rescaling, policy_preprocessor, policy_classifier]

    # Define cost and gradient update op
    a_rescaling = tf.placeholder("float", [None, rescaling_num]) # mask vector for action
    a_preprocessor = tf.placeholder("float", [None, preprocessor_num]) # mask vector for action
    a_classifier = tf.placeholder("float", [None, classifier_num]) # mask vector for action
    a = [a_rescaling, a_preprocessor, a_classifier]
    y = tf.placeholder("float", [None]) # reward
    action_pred_values = 0
    for i in range(3):
        action_pred_values += tf.reduce_sum(tf.multiply(policy_values[i], a[i]), reduction_indices=1)

    l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in tf.trainable_variables()  ])
    pg_loss = tf.reduce_mean(-tf.log(action_pred_values)*y)

    cost = pg_loss + 0.002 * l2_loss
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grad_update = optimizer.minimize(cost, var_list=network_params)

    graph_ops = {"s" : s,
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
    logged_epsilon = tf.Variable(0.)
    tf.summary.scalar("Epsilon", logged_epsilon)
    logged_T = tf.Variable(0.)
    summary_vars = [episode_reward, logged_epsilon]
    summary_placeholders = [tf.placeholder("float") for i in range(len(summary_vars))]
    update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
    summary_op = tf.summary.merge_all()
    return summary_placeholders, update_ops, summary_op


def train(session, graph_ops, num_actions, saver):
    # Init environment (default load all training dataset)
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
    policy_values = graph_ops["policy_values"]

    for i_episode in range(FLAGS.num_eval_episodes):
        ep_reward = []
        batch_size = 1
        for b in range(int(env.size/batch_size)):
            s_t = env.data_state(b, batch_size)
            policy_rescaling, policy_preprocessor, policy_classifier = policy_values[0], policy_values[1], policy_values[2]
            outrescaling_t = policy_rescaling.eval(session = session, feed_dict = {s : s_t})
            outpreprocessor_t = policy_preprocessor.eval(session = session, feed_dict = {s : s_t})
            outclassifier_t = policy_classifier.eval(session = session, feed_dict = {s : s_t})

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

    input_shape, rescaling_num, preprocessor_num, classifier_num = 29, 6, 13, 15
    num_actions = [rescaling_num, preprocessor_num, classifier_num]
    graph_ops = build_graph(input_shape, num_actions)
    saver = tf.train.Saver()

    if FLAGS.testing:
        evaluation(session, graph_ops, saver)
    else:
        train(session, graph_ops, num_actions, saver)

if __name__ == "__main__":
    main()
