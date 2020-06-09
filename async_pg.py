#!/usr/bin/env python
import os

import threading
import tensorflow as tf
import sys
import random
import numpy as np
import time
from env import Env
from model import build_network

flags = tf.app.flags

flags.DEFINE_string('experiment', 'async_dqn', 'Name of the current experiment')
flags.DEFINE_string('dataset', '6', 'Name of dataset to run')
flags.DEFINE_integer('num_concurrent', 1, 'Number of concurrent actor-learner threads to use during training.')
flags.DEFINE_integer('tmax', 100, 'Number of training timesteps.')
flags.DEFINE_integer('agent_history_length', 4, 'Use this number of recent screens as the environment state.')
flags.DEFINE_integer('network_update_frequency', 10, 'Frequency with which each actor learner thread does an async gradient update')
flags.DEFINE_integer('target_network_update_frequency', 25, 'Reset the target network every n timesteps')
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_float('gamma', 0.99, 'Reward discount rate.')
flags.DEFINE_integer('anneal_epsilon_timesteps', 10, 'Number of timesteps to anneal epsilon.')
flags.DEFINE_string('summary_dir', './pg/summaries', 'Directory for storing tensorboard summaries')
flags.DEFINE_string('checkpoint_dir', './pg/checkpoints', 'Directory for storing model checkpoints')
flags.DEFINE_integer('summary_interval', 20,
                     'Save training summary to file every n seconds (rounded '
                     'up to statistics interval.')
flags.DEFINE_integer('checkpoint_interval', 10,
                     'Checkpoint the model (i.e. save the parameters) every n '
                     'seconds (rounded up to statistics interval.')
flags.DEFINE_boolean('show_training', True, 'If true, have gym render evironments during training')
flags.DEFINE_boolean('testing', False, 'If true, run gym evaluation')
flags.DEFINE_string('checkpoint_path', './pg/checkpoints/async_pg.ckpt-100', 'Path to recent checkpoint to use for evaluation')
flags.DEFINE_string('eval_dir', './pg/', 'Directory to store gym evaluation')
flags.DEFINE_integer('num_eval_episodes', 1, 'Number of episodes to run env evaluation.')
FLAGS = flags.FLAGS
T = 0
TMAX = FLAGS.tmax

def sample_final_epsilon():
    """
    Sample a final epsilon value to anneal towards from a distribution.
    These values are specified in section 5.1 of http://arxiv.org/pdf/1602.01783v1.pdf
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

def actor_learner_thread(thread_id, env, session, graph_ops, num_actions, summary_ops, saver):
    """
    Actor-learner thread implementing asynchronous Policy Gradient, as specified
    in algorithm 1 here: http://arxiv.org/pdf/1602.01783v1.pdf.
    """
    global TMAX, T

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

    print("Starting thread ", thread_id, "with final epsilon ", final_epsilon)

    time.sleep(3*thread_id)
    t = 0
    max_reward = 0
    batch_size = 2
    while T < TMAX:
        for b in range(int(env.size/batch_size)):
            # Get initial game observation
            s_t = env.data_state(b, batch_size)
            terminal = False

            # Set up per-episode counters
            ep_reward = 0
            episode_ave_max_q = 0
            ep_t = 0

            while True:
                # Forward the deep q network, get Q(s,a) values
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
        
                # Action execution on behalf of actor-learner
                s_t1, r_t, terminal, info = env.run_actions((rescaling_action, preprocessor_action, classifier_action), b, batch_size)
                #print(r_t)

                #clipped_r_t = np.clip(r_t, 0, 1)
                y_batch.append(-r_t)
                y_batch = np.reshape(np.array(y_batch), [-1])
                print(y_batch.shape)
        
                rescaling_batch.append(rescaling_t)
                rescaling_batch = np.reshape(np.array(rescaling_batch), [-1, np.array(rescaling_t).shape[1]])
                preprocessor_batch.append(preprocessor_t)
                preprocessor_batch = np.reshape(np.array(preprocessor_batch), [-1, np.array(preprocessor_t).shape[1]])
                classifier_batch.append(classifier_t)
                classifier_batch = np.reshape(np.array(classifier_batch), [-1, np.array(classifier_t).shape[1]])

                s_batch.append(s_t)
                s_batch = np.reshape(np.array(s_batch), [-1, np.array(s_t).shape[1]])
                print(s_batch.shape)
                
        
                # Update the state and counters
                s_t = s_t1
                T += 1
                t += 1

                ep_t += 1
                ep_reward += r_t
                episode_ave_max_q += np.max(outrescaling_t) + np.max(outpreprocessor_t) + np.max(outclassifier_t)

                print("T=",T,", t=",t)

                # Save model progress
                if t % FLAGS.checkpoint_interval == 0:
                #if r_t > max_reward:
                    #max_reward = r_t
                    print("==========save(session), reward=", r_t)
                    saver.save(session, FLAGS.checkpoint_dir+"/"+FLAGS.experiment+".ckpt", global_step = t)

                loss = 0
                #if t % FLAGS.network_update_frequency == 0:
                if len(s_batch) > 0:
                    print("===========run(grad_update)")
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
        
                # Print end of episode stats
                #stats = [ep_reward, episode_ave_max_q/float(ep_t), epsilon]
                #for i in range(len(stats)):
                #    session.run(update_ops[i], feed_dict={summary_placeholders[i]:float(stats[i])})
                print("THREAD:", thread_id, "/ TIME", T, "/ TIMESTEP", t, "/ EPSILON", epsilon, "/ REWARD", ep_reward, 
                    "/ AVE_MAX %.4f" % (episode_ave_max_q/float(ep_t)), "/ EPSILON PROGRESS", t/float(FLAGS.anneal_epsilon_timesteps), "/LOSS", loss)
                break

def copy_src_to_dst(from_scope, to_scope):
    """Creates a copy variable weights operation
    Returns:
        list: Each element is a copy operation
    """
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def build_graph(input_shape, num_actions):
    rescaling_num, preprocessor_num, classifier_num = num_actions[0], num_actions[1], num_actions[2]
    # Create policy network
    s, policy_rescaling, policy_preprocessor, policy_classifier = build_network(input_shape, rescaling_num, preprocessor_num, classifier_num, name="policy_network")
    network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy_network")
    policy_values = [policy_rescaling, policy_preprocessor, policy_classifier]

    # Op for periodically updating target network with online network weights
    reset_target_network_params = []
    
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
    # Initialize target network weights
    #reset_target_network_params = copy_src_to_dst("q_network", "target_q_network")
    #session.run(reset_target_network_params)

    # Set up environments (one per thread)
    envs = [Env() for i in range(FLAGS.num_concurrent)]
    
    summary_ops = setup_summaries()
    summary_op = summary_ops[-1]

    # Initialize variables
    session.run(tf.initialize_all_variables())
    summary_save_path = FLAGS.summary_dir + "/" + FLAGS.experiment
    writer = tf.summary.FileWriter(summary_save_path, session.graph)
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    # Start num_concurrent actor-learner training threads
    actor_learner_threads = [threading.Thread(target=actor_learner_thread, args=(thread_id, envs[thread_id], session, graph_ops, num_actions, summary_ops, saver)) 
                            for thread_id in range(FLAGS.num_concurrent)]
    for t in actor_learner_threads:
        t.start()

    # Show the agents training and write summary statistics
    last_summary_time = 0
    for t in actor_learner_threads:
        t.join()

def evaluation(session, graph_ops, saver):
    saver.restore(session, FLAGS.checkpoint_path)
    print("Restored model weights from ", FLAGS.checkpoint_path)
    env = Env(FLAGS.dataset)

    # Unpack graph ops
    s = graph_ops["s"]
    policy_values = graph_ops["policy_values"]

    for i_episode in range(FLAGS.num_eval_episodes):
        batch_idx = 0
        batch_size = 1
        s_t = env.data_state(batch_idx, batch_size)
        ep_reward = 0
        terminal = False
        #while not terminal:
        policy_rescaling, policy_preprocessor, policy_classifier = policy_values[0], policy_values[1], policy_values[2]
        outrescaling_t = policy_rescaling.eval(session = session, feed_dict = {s : [s_t]})
        outpreprocessor_t = policy_preprocessor.eval(session = session, feed_dict = {s : [s_t]})
        outclassifier_t = policy_classifier.eval(session = session, feed_dict = {s : [s_t]})

        rescaling_action = np.argmax(outrescaling_t)
        preprocessor_action = np.argmax(outpreprocessor_t)
        classifier_action = np.argmax(outclassifier_t)

        s_t1, r_t, terminal, info = env.run_actions((rescaling_action, preprocessor_action, classifier_action), batch_idx, batch_size)
        s_t = s_t1
        ep_reward += r_t
        print(ep_reward)

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
