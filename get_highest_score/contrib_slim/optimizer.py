import tensorflow as tf

def basicOptimizer():
    return tf.train.GradientDescentOptimizer(0.01)
