"""
Test file for input queue:
usage
python test_input_queue.py
"""
import tensorflow as tf
import sys
import os

# patch for import
sys.path += [os.path.join(os.getcwd(),"model")]

from model.notmnist_input import distorted_inputs

# define graph


tf.flags.DEFINE_integer('train_batch_size', 32, 'Batch size for training.')

tf.flags.DEFINE_integer('eval_batch_size', 100, 'Batch size for validation.')
tf.flags.DEFINE_string("data_dir","model","folder for data")

# weights = tf.Variable(tf.truncated_normal(shape=[IMAGE_WIDTH*IMAGE_HEIGHT, NUM_CLASS]))

FLAGS = tf.flags.FLAGS

def main(unused_args):
    image,label = distorted_inputs()

    print("start run")
    sess = tf.Session()
    tf.train.start_queue_runners(sess=sess)
    sess.run(tf.global_variables_initializer())
    all_pass = True
    for i in range(10):
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        image_run,label_run = sess.run([image,label])
        all_pass = all_pass and (image_run.shape == (FLAGS.train_batch_size, 28, 28, 1) ) and \
                   (label_run.shape == (FLAGS.train_batch_size,))
    print("*"*30)
    if all_pass:
        print("Test Pass")
    else:
        print("Test Failed")

if __name__ == "__main__":
    tf.app.run()