# python importings
import functools

# tensorflow importinges
import tensorflow as tf

FLAGS = tf.flags.FLAGS

# my modules
from notmnist import variable_on_cpu, variable_on_cpu_wd,variable_summaries
from notmnist_input import NUM_CLASS,IMAGE_HEIGHT


def model_fn_cnn(features, labels, mode):
    with tf.device("/gpu:0"):
        with tf.name_scope("conv1"):
            conv1_patch_size = 16
            conv1_weights = variable_on_cpu_wd("weights",
                                            initializer=tf.truncated_normal(
                                                shape=[2,2,1, conv1_patch_size],
                                                dtype=tf.float32,
                                                stddev=0.1),
                                                wd=FLAGS.weight_decay)
            conv1_biases = variable_on_cpu("biases",
                                           initializer=tf.constant(
                                               shape=[conv1_patch_size],
                                               dtype=tf.float32,
                                               value=0))
            conv = tf.nn.conv2d(features,
                                conv1_weights,
                                strides=[1,2,2,1],
                                padding='SAME')
            pre_activation = tf.nn.bias_add(
                conv,
                conv1_biases)
            conv1 = tf.nn.relu(pre_activation)
            variable_summaries(conv1_weights,"weights")
            variable_summaries(conv1_biases,"biases")

        with tf.name_scope("pool1"):
            pool1 = tf.nn.max_pool(conv1,
                                  ksize=[1,2,2,1],
                                  strides=[1,2,2,1],
                                  padding='SAME')

        with tf.name_scope("fc1") as scope:
            feature_size = [i.value for i in pool1.shape]
            feature_size = functools.reduce(lambda x,y: x*y, feature_size[1:])
            reshape = tf.reshape(pool1,[-1,feature_size])
            dim = reshape.shape[1].value
            fc1_nodes = 128
            fc1_weights = variable_on_cpu_wd("weights",
                                          initializer=tf.truncated_normal(
                                              shape=[dim,fc1_nodes],
                                              dtype=tf.float32,
                                              stddev=0.1),
                                             wd=FLAGS.weight_decay)
            fc1_biases = variable_on_cpu("biases",
                                         initializer=tf.constant(
                                             value=0,
                                             shape=[fc1_nodes],
                                             dtype=tf.float32))
            fc1 = tf.nn.relu(tf.add(
                                tf.matmul(reshape,fc1_weights),
                                fc1_biases))
            variable_summaries(fc1_weights,"weights")
            variable_summaries(fc1_biases,"biases")

        with tf.name_scope("softmax_linear"):
            sf_weights = variable_on_cpu_wd("weights",
                                         initializer=tf.truncated_normal(
                                             shape=[fc1_nodes,NUM_CLASS],
                                             dtype=tf.float32,
                                             stddev=0.1),
                                            wd=FLAGS.weight_decay
                                         )
            sf_biases = variable_on_cpu("biases",
                                        initializer=tf.constant(
                                            value=0,
                                            shape=[10],
                                            dtype=tf.float32))
            logits = tf.add(tf.matmul(fc1,sf_weights),sf_biases)
            variable_summaries(sf_weights, "weights")
            variable_summaries(sf_biases, "biases")
        # import  pdb;pdb.set_trace()
        with tf.name_scope("loss"):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            loss_rm = tf.reduce_mean(loss)
            tf.add_to_collection("losses",loss_rm)
            total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
            tf.add_to_collection("total_losses",total_loss)
            tf.summary.scalar("total_loss", total_loss)

        with tf.name_scope("optimizer"):
            global_step = tf.train.get_global_step()
            starter_learning_rate = 0.1
            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                       100000, 0.96, staircase=True)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            tf.summary.scalar("learning_rate",learning_rate)
            # train_op = optimizer.minimize(loss_rm,global_step=tf.train.get_global_step())
            grads = optimizer.compute_gradients(total_loss)
            # Apply gradients.
            apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
            for grad, var in grads:
                if grad is not None:
                    tf.summary.histogram(var.op.name + '/gradients', grad)

        with tf.name_scope("prediction"):
            predictions = {
                "classes":tf.argmax(input=logits,axis=1),
                "probabilities":tf.nn.softmax(logits)
            }

        stacked_labels = tf.concat(labels, axis=0)
        metrics = {
            'accuracy': tf.metrics.accuracy(stacked_labels, predictions['classes'])
        }

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=total_loss,
        train_op=apply_gradient_op,
        eval_metric_ops=metrics
        )


def model_fn_fc(features, labels, mode):
    """
    implement a basic model with fully connected layer.
    hiden layer: 128
    :param features:
    :param labels:
    :param mode:
    :return:
    """
    with tf.device("/gpu:0"):
        with tf.name_scope("fc1") as scope:
            feature_size = [i.value for i in features.shape]
            feature_size = functools.reduce(lambda x,y: x*y, feature_size[1:])
            reshape = tf.reshape(features,[-1,feature_size])
            dim = reshape.shape[1].value
            fc1_output_num = 128
            fc1_weights = variable_on_cpu("weights",
                        initializer=tf.truncated_normal(shape=[dim, fc1_output_num],dtype=tf.float32))
            fc1_biases = variable_on_cpu("biases",
                        initializer=tf.constant(value=0,shape=[fc1_output_num], dtype=tf.float32))
            fc1 = tf.nn.relu(tf.matmul(reshape, fc1_weights)+fc1_biases)
            variable_summaries(fc1_weights, "weights")
            variable_summaries(fc1_biases, "biases")

        with tf.name_scope("softmax_linear"):
            sf_weights = variable_on_cpu("weights",
                                         initializer=tf.truncated_normal(
                                             shape=[fc1_output_num, NUM_CLASS],
                                             dtype=tf.float32)
                                         )
            sf_biases = variable_on_cpu("biases",
                                        initializer=tf.constant(
                                            value=0,
                                            shape=[NUM_CLASS],
                                            dtype=tf.float32))
            logits = tf.add(tf.matmul(fc1,sf_weights),sf_biases)
            variable_summaries(sf_weights, "weights")
            variable_summaries(sf_biases, "biases")
        with tf.name_scope("loss"):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            loss_rm = tf.reduce_mean(loss)
            tf.summary.scalar("loss", loss_rm)

        with tf.name_scope("optimizer"):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
            grads = optimizer.compute_gradients(loss_rm)
            apply_gradient_op = optimizer.apply_gradients(grads, global_step=tf.train.get_global_step())
            # write grad to summary as histogram
            for grad, var in grads:
                if grad is not None:
                    tf.summary.histogram(var.op.name + '/gradients', grad)

        with tf.name_scope("prediction"):
            predictions = {
                "classes":tf.argmax(input=logits,axis=1),
                "probabilities":tf.nn.softmax(logits)
            }

        stacked_labels = tf.concat(labels, axis=0)
        metrics = {
            'accuracy': tf.metrics.accuracy(stacked_labels, predictions['classes'])
        }

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss_rm,
        train_op=apply_gradient_op,
        eval_metric_ops=metrics
        )