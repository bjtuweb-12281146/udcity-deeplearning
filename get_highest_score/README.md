The target of this example is try to get higher accuracy of nonMNIST dataset. the dataset and some of code is come from [this course](https://classroom.udacity.com/courses/ud730)

## Code structure

### Create tfrecords for data.
nonmnist_data.py

prepare the dataset for this task




## Knowlege point

## tf.train.Estimator 
* model_fn => define a full-feature model_fn    
    *tf.train.EstimatorSpec
    
## [input_fn](https://tensorflow.google.cn/get_started/input_fn)

* implement a full-feature for training, evaluation, predict.
* put data process in to it.


### commit: 1c5da664f0d117cb64a8f5a95c73245bc3dcdc8e
test data:{'accuracy': 0.89953125, 'loss': 3.6034086, 'global_step': 4000000}
training data:{'accuracy': 0.98312497, 'loss': 0.048656259, 'global_step': 4000000}

### L2: 0.01
Evaluation on test data...
{'loss': 0.6000461, 'accuracy': 0.86624998, 'global_step': 2873366}
Evaluation on training data...
{'loss': 0.59285873, 'accuracy': 0.87085938, 'global_step': 2873366}


