import tensorflow as tf
import os
import pdb


from notmnist_data import train_tfrecords_name,valid_tfrecords_name

FLAGS = tf.flags.FLAGS

IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
NUM_CLASS = 10


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialize_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialize_example,
        features={

            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string)
            }
        )
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image,[IMAGE_HEIGHT,IMAGE_WIDTH,1])
    image = tf.cast(image,tf.float32)

    label = tf.cast(features["label"],tf.int32)
    label = tf.reshape(label,[])

    return image,label


def distorted_inputs(is_training=True):
    if is_training:
        filename_queue = tf.train.string_input_producer([os.path.join(FLAGS.data_dir, train_tfrecords_name)])
    else:
        filename_queue = tf.train.string_input_producer([os.path.join(FLAGS.data_dir, valid_tfrecords_name)])

    image,label = read_and_decode(filename_queue)
    float_image = tf.image.per_image_standardization(image)

    # pdb.set_trace()
    images,labels = tf.train.shuffle_batch(
        [float_image,label], batch_size=FLAGS.train_batch_size,num_threads=FLAGS.input_data_thread,
        capacity=8196+FLAGS.input_data_thread*FLAGS.train_batch_size,min_after_dequeue=8196
        )
    tf.summary.image("training_image", images)
    return images,labels


def input_fn(subset):
    is_training = subset == "training"
    with tf.name_scope("input_data"):
        return distorted_inputs(is_training)
