import unittest
from model.notmnist_data  import make_tfrecords,convert_back
import numpy as np


class TestDataConversion(unittest.TestCase):
    def test_dataintegrity(self):
        train_data,train_label,valid_data,valid_label,\
        test_data, test_label = make_tfrecords(sample_num=3341,force=True)
        data_cb,label_cb = convert_back('train_data.tfrecords')
        valid_data_cb,valid_label_cb = convert_back('valid_data.tfrecords')
        test_data_cb,test_label_cb = convert_back('test_data.tfrecords')
        all_same = np.allclose(train_data,data_cb) and np.allclose(train_label,label_cb) and \
            np.allclose(valid_data,valid_data_cb) and np.allclose(valid_label,valid_label_cb) and \
            np.allclose(test_data,test_data_cb) and np.allclose(test_label,test_label_cb)
        self.assertTrue(all_same)
