import tensorflow as tf

from not_mnist_helper import get_datasets, reformat
from data_helper import accuracy
from constants import *


def get_reformatted_datasets():
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = get_datasets()
    train_dataset, train_labels = reformat(train_dataset, train_labels)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
    test_dataset, test_labels = reformat(test_dataset, test_labels)
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)
    return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels


def set_tf_variables_for_gradient_descent(graph, train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels):
    with graph.as_default():
        # Input data.
        # Load the training, validation and test data into constants that are
        # attached to the graph.
        tf_train_dataset = tf.constant(train_dataset)
        tf_train_labels = tf.constant(train_labels)
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)
        weight = tf.Variable(tf.truncated_normal(
            [IMAGE_SIZE * IMAGE_SIZE, NUM_OF_CLASSES]))
        biases = tf.Variable(tf.zeros([NUM_OF_CLASSES]))
        # Training computation.
        # We multiply the inputs with the weight matrix, and add biases. We compute
        # the softmax and cross-entropy (it's one operation in TensorFlow, because
        # it's very common, and it can be optimized). We take the average of this
        # cross-entropy across all training examples: that's our loss.
        logits = tf.matmul(train_dataset, weight) + biases
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=train_labels, logits=logits))

        # Optimizer.
        # We are going to find the minimum of this loss using gradient descent.
        optimizer = tf.train.GradientDescentOptimizer(
            LEARNING_RATE).minimize(loss)

        # Predictions for the training, validation, and test data.
        # These are not part of training, but merely here so that we can report
        # accuracy figures as we train.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(
            tf.matmul(tf_valid_dataset, weight) + biases)
        test_prediction = tf.nn.softmax(
            tf.matmul(tf_test_dataset, weight) + biases)
        return optimizer, loss, train_prediction, valid_prediction, test_prediction


def run_gradient_descent(graph, optimizer, loss, train_prediction, train_labels, valid_prediction, valid_labels, test_prediction, test_labels):
    num_of_steps = 801
    with tf.Session(graph=graph) as session:
        # This is a one-time operation which ensures the parameters get initialized as
        # we described in the graph: random weights for the matrix, zeros for the
        # biases.
        tf.global_variables_initializer().run()
        print('Initialized')
        for step in range(num_of_steps):
            # Run the computations. We tell .run() that we want to run the optimizer,
            # and get the loss value and the training predictions returned as numpy
            # arrays.
            _, l, predictions = session.run(
                [optimizer, loss, train_prediction])
            if (step % 100 == 0):
                print('-----------------------------------------------------')
                print('Loss at step %d: %f' % (step, l))
                print('Training accuracy: %.1f%%' % accuracy(
                    predictions, train_labels))
                # Calling .eval() on valid_prediction is basically like calling run(), but
                # just to get that one numpy array. Note that it recomputes all its graph
                # dependencies.
                print('Validation accuracy: %.1f%%' %
                      accuracy(valid_prediction.eval(), valid_labels))
                print('-----------------------------------------------------')
        print('Test accuracy: %.1f%%' % accuracy(
            test_prediction.eval(), test_labels))


def set_tf_variables_for_stochastic_gradient_descent(graph, valid_dataset, test_dataset):
    with graph.as_default():
        # Input data. For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
        tf_train_dataset = tf.placeholder(
            tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE*IMAGE_SIZE))
        tf_train_labels = tf.placeholder(
            tf.float32, shape=(BATCH_SIZE, NUM_OF_CLASSES))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        # Variables
        weights = tf.Variable(tf.truncated_normal(
            [IMAGE_SIZE*IMAGE_SIZE, NUM_OF_CLASSES]))
        biases = tf.Variable(tf.zeros([NUM_OF_CLASSES]))

        # Training computation
        logits = tf.matmul(tf_train_dataset, weights) + biases
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=tf_train_labels, logits=logits))

        # Optimiser
        optimizer = tf.train.GradientDescentOptimizer(
            LEARNING_RATE).minimize(loss)

        # Predictions for the training, validation and test data.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(
            tf.matmul(tf_valid_dataset, weights)+biases)
        test_prediction = tf.nn.softmax(
            tf.matmul(tf_test_dataset, weights)+biases)


def run_stochastic_gradient_descent():
    num_of_steps = 3001
    pass


def main():
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = get_reformatted_datasets()
    # Gradient descent
    # gradient_descent_graph = tf.Graph()

    # optimizer, loss, train_prediction, valid_prediction, test_prediction = set_tf_variables_for_gradient_descent(
    #     gradient_descent_graph, train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels)

    # run_gradient_descent(gradient_descent_graph, optimizer, loss, train_prediction, train_labels,
    #                      valid_prediction, valid_labels, test_prediction, test_labels)

    # Stochastic Gradient descent
    stochastic_gradient_descent_graph = tf.Graph()

    set_tf_variables_for_stochastic_gradient_descent(
        stochastic_gradient_descent_graph, valid_dataset, test_dataset)

    run_stochastic_gradient_descent()


if __name__ == '__main__':
    main()
