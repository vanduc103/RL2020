import tensorflow as tf

def build_network(input_shape=29, rescaling_num=6, preprocessor_num=13, classifier_num=15, name="network"):
    with tf.variable_scope(name):
        states = tf.placeholder(tf.float32, shape=[None, input_shape], name="states")

        net = states
        with tf.variable_scope("layer1"):
            net = tf.nn.relu(tf.layers.dense(net, 32, name='hidden1'))
        with tf.variable_scope("layer2"):
            net = tf.nn.relu(tf.layers.dense(net, 32, name='hidden2'))

        # rescaling - data preprocessing
        rescaling = tf.layers.dense(net, rescaling_num, name='rescaling')
        rescaling_prob = tf.nn.softmax(rescaling)

        # preprocessor
        preprocessor = tf.layers.dense(net, preprocessor_num, name='preprocessor')
        preprocessor_prob = tf.nn.softmax(preprocessor)

        # classifier
        classifier = tf.layers.dense(net, classifier_num, name='classifier')
        classifier_prob = tf.nn.softmax(classifier)

    return states, rescaling_prob, preprocessor_prob, classifier_prob
