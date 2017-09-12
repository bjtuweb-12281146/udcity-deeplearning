# basic python library
import functools
import logging
import pdb
import datetime

# tensorflow importings
import tensorflow as tf
from tensorflow.contrib.learn.python.learn import estimators as contrib_estimator

# my modules
from notmnist_model import model_fn_fc,model_fn_cnn, get_experiment_fn
from notmnist_input import input_fn
from notmnist import ExamplesPerSecondHook

tf.flags.DEFINE_integer('train_batch_size', 128, 'Batch size for training.')
tf.flags.DEFINE_integer('eval_batch_size', 100, 'Batch size for validation.')
tf.flags.DEFINE_integer('train_steps', 10000, 'Train step for this training')
tf.flags.DEFINE_integer('input_data_thread', 8, 'Batch size for validation.')
tf.flags.DEFINE_float('weight_decay',0.001,"l2 regularization")
tf.flags.DEFINE_boolean('log_device_placement', False,'Whether to log device placement.')
tf.flags.DEFINE_string("data_dir","","folder for data")
tf.flags.DEFINE_string("train_dir","/tmp/mymnist","folder for checkpoint file")
tf.flags.DEFINE_boolean('run_experiment', True,'if use Experiment to run the training')

from tensorflow.contrib.learn import monitors

FLAGS = tf.flags.FLAGS


def main(unused_argv):
    tf.logging.set_verbosity(logging.ERROR)

    # configuration for model
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)
    sess_config.allow_soft_placement = True
    sess_config.log_device_placement = FLAGS.log_device_placement

    if FLAGS.run_experiment:
        config = tf.contrib.learn.RunConfig(model_dir=FLAGS.train_dir)
        # config = config.replace(sess_config=sess_config)
        tf.logging.set_verbosity(logging.INFO)
        validation_monitor = monitors.ValidationMonitor(
            input_fn=functools.partial(input_fn, subset="evaluation"),
            eval_steps=128,
            every_n_steps=101,
            early_stopping_metric="accuracy",
            early_stopping_rounds = 1000
        )
        run_monitors = [validation_monitor]
        tf.contrib.learn.learn_runner.run(
            get_experiment_fn(train_input_fn=functools.partial(input_fn,subset="training"),
                              eval_input_fn=functools.partial(input_fn,subset="evaluation"),
                              train_steps=FLAGS.train_steps,
                              eval_steps=100,
                              train_hooks=run_monitors,
                              model_fn=model_fn_cnn
                              ),
            run_config = config
        )
    else:
        start_time = datetime.datetime.now()
        config = tf.estimator.RunConfig()
        config = config.replace(session_config=sess_config)
        per_example_hook = ExamplesPerSecondHook(FLAGS.train_batch_size, every_n_steps=100)
        hooks = [per_example_hook]
        classifier = tf.estimator.Estimator(
            model_fn=model_fn_cnn,
            model_dir= FLAGS.train_dir,
            config=config
        )
        classifier.train(input_fn=functools.partial(input_fn,subset="training"),
                         steps=FLAGS.train_steps,
                         hooks=hooks
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