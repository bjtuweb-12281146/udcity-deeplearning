import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.nets import alexnet
from utils import notmnist_input

def logits(inputs,
           num_classes=notmnist_input.NUM_CLASSES,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='alexnet_v2'):
    """
    inputs: a tensor of size [batch_size, height, width, channels]
    """
    with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
        outputs, _ = alexnet.alexnet_v2(inputs)
    return outputs


def loss(logits, label):
    """
    logits: output of network, shape=[batch_size,num_class]
    label: ground true of training data: shape=[1,]
    """
    return slim.losses.sparse_softmax_cross_entropy(logits,label)
