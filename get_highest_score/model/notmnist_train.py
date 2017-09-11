# basic python library
import functools
import logging
import pdb
import datetime

# tensorflow importings
import tensorflow as tf

# my modules
from notmnist_model import model_fn_fc,model_fn_cnn
from notmnist_input import input_fn
from notmnist import ExamplesPerSecondHook

tf.flags.DEFINE_integer('train_batch_size', 128, 'Batch size for training.')
tf.flags.DEFINE_integer('eval_batch_size', 100, 'Batch size for validation.')
tf.flags.DEFINE_integer('train_steps', 10000, 'Train step for this training')
tf.flags.DEFINE_integer('input_data_thread', 8, 'Batch size for validation.')
tf.flags.DEFINE_float('weight_decay',0.001,"l2 regularization")
tf.flags.DEFINE_boolean('log_device_placement', False,'Whether to log device placement.')
tf.flags.DEFINE_string("data_dir","","folder for data")
tf.flags.DEFINE_string("train_dir","/temp/mymnist","folder for checkpoint file")


FLAGS = tf.flags.FLAGS


def main(unused_argv):
    tf.logging.set_verbosity(logging.INFO)

    # configuration for model
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)
    sess_config.allow_soft_placement = True
    sess_config.log_device_placement = FLAGS.log_device_placement

    config = tf.estimator.RunConfig()
    config = config.replace(session_config=sess_config)
    # per_example_hook = ExamplesPerSecondHook(every_n_steps=10000)
    # hooks = [per_example_hook]
    classifier = tf.estimator.Estimator(
        model_fn=model_fn_cnn,
        model_dir= FLAGS.train_dir,
        config=config,
        # hooks=hooks
    )
    print("start to train...")
    start_time = datetime.datetime.now()
    classifier.train(input_fn=functools.partial(input_fn,subset="training"),
                     steps=FLAGS.train_steps
                     )

    train_time = datetime.datetime.now() - start_time
    print("Training complete in : minutes:{}".format(train_time.total_seconds()/60))
    tf.logging.set_verbosity(logging.WARNING)
    print('Evaluation on test data...')
    eval_results = classifier.evaluate(
        input_fn=functools.partial(input_fn,subset="evaluation"),
        steps=100)
    print(eval_results)

    print("Evaluation on training data...")
    eval_results = classifier.evaluate(
        input_fn=functools.partial(input_fn,subset="training"),
        steps=100)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()