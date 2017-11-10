import tensorflow as tf
from . import notmnist, optimizer
import tensorflow.contrib.slim as slim
from utils import notmnist_input

tf.app.flags.DEFINE_string('train_dir', '/tmp/notmnist_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                           """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/home/scott/temp/notmnistdata',
                          """Path to the CIFAR-10 data directory.""")

FLAGS=tf.app.flags.FLAGS

def main(argv=None):
    with tf.device("/cpu:0"):
        inputs,labels=notmnist_input.distorted_inputs(FLAGS.data_dir, FLAGS.batch_size)
    logits=notmnist.logits(inputs)
    loss=notmnist.loss(logits, labels)
    optimizer=optimizer.basicOptimizer()
    train_op=slim.learning.create_train_op(loss,optimizer)
    slim.learning.train(train_op,FLAGS.train_dir)

if __name__ == "__main__":
    tf.app.run()
