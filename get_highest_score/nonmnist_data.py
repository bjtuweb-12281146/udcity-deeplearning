import os
from six.moves.urllib.request import urlretrieve
import sys
import tarfile
# import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import tensorflow as tf


url = 'https://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None
data_root = '.' # Change me to store data elsewhere

def download_progress_hook(count, blockSize, totalSize):
  """A hook to report the progress of a download. This is mostly intended for users with
  slow internet connections. Reports every 5% change in download progress.
  """
  global last_percent_reported
  percent = int(count * blockSize * 100 / totalSize)

  if last_percent_reported != percent:
    if percent % 5 == 0:
      sys.stdout.write("%s%%" % percent)
      sys.stdout.flush()
    else:
      sys.stdout.write(".")
      sys.stdout.flush()

    last_percent_reported = percent

def maybe_download(filename, expected_bytes, force=False):
  """Download a file if not present, and make sure it's the right size."""
  dest_filename = os.path.join(data_root, filename)
  if force or not os.path.exists(dest_filename):
    print('Attempting to download:', filename)
    filename, _ = urlretrieve(url + filename, dest_filename, reporthook=download_progress_hook)
    print('\nDownload Complete!')
  statinfo = os.stat(dest_filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', dest_filename)
  else:
    raise Exception(
      'Failed to verify ' + dest_filename + '. Can you get to it with a browser?')
  return dest_filename


num_classes = 10
# np.random.seed(133)

def maybe_extract(filename, force=False):
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  if os.path.isdir(root) and not force:
    # You may override by setting force=True.
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall(data_root)
    tar.close()
  data_folders = [
    os.path.join(root, d) for d in sorted(os.listdir(root))
    if os.path.isdir(os.path.join(root, d))]
  if len(data_folders) != num_classes:
    raise Exception(
      'Expected %d folders, one per class. Found %d instead.' % (
        num_classes, len(data_folders)))
  print(data_folders)
  return data_folders


# build ndarray to store image and label

def load_letter(folder,label,image_size=28,sample_num=-1):
  """Load the data for a single letter label."""
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
  num_images = 0
  if sample_num == -1:
      sample_num = len(image_files)
  for image in image_files:
    image_file = os.path.join(folder, image)
    try:
      image_data = ndimage.imread(image_file).astype(float)
      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[num_images, :, :] = image_data
      num_images = num_images + 1
      if num_images >= sample_num:
          break
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

  dataset = dataset[0:num_images, :, :]
  data_label = np.ndarray(shape=(num_images), dtype=np.int8)
  data_label.fill(label)
  return dataset,data_label


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(data_set, data_set_label, name):
  """
  args:
  data_set: a ndarray with following shape: [data_set_size,height,width]
  data_set_lable: a ndarray with following shape: [data_set_size]
  name: the name of generated tfrecoords file: name.tfrecoords
  return:
    none
  """
  if data_set.shape[0] != data_set_label.shape[0]:
    raise ValueError('Images size %d does not match label size %d.' %
                     (data_set.shape[0], data_set_label.shape[0]))

  num_examples = data_set.shape[0]
  rows = data_set.shape[1]
  cols = data_set.shape[2]

  filename = name + '.tfrecords'
  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  for index in range(num_examples):
    image_raw = data_set[index].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(rows),
        'width': _int64_feature(cols),
        'label': _int64_feature(data_set_label[index]),
        'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())
  writer.close()


def convert_back(file_name)
    """
    target:
    convert a tfrecods without difining a graph. for test purpose
    args:
        file_name: file name of tfrecods file.abs
    return:
        (reconstructed_image,reconstructed_label)
    """
    constructed_data = None
    constructed_label = None
    record_iterator = tf.python_io.tf_record_iterator(path=file_name)

    for record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(record)
        height = int(example.features['height'].int64_list.value[0])
        width = int(example.features['width'].int64_list.value[0])
        label = int(example.features['label'].int64_list.value[0])
        image_raw = example.features['image_raw'].bytes_list.value[0]

        img_1d = np.fromstring(image_raw,dtype=np.uint8)
        img = img_1d.reshape((height,width))
        if constructed_data == None:
            constructed_data = img
            constructed_label = np.array(label)
        else:
            np.concatenate([constructed_data,img])
            np.append(constructed_label,label)
    return constructed_data,constructed_label


def make_tfrecords():
    # Download data
    train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
    test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)
    # extract data to folder
    train_folders = maybe_extract(train_filename)
    test_folders = maybe_extract(test_filename)

    train_data_list = []
    for file_folder in train_folders:
        data,label = load_letter(file_folder, ord(os.path.basename(file_folder))-ord('A') )
        train_data_list.append((data,label))
        # concatenate the whole list and shuffle it.
    train_dataset = np.concatenate([ train_data_list[i][0] for i in range(len(train_data_list))])
    train_label = np.concatenate([train_data_list[i][1] for i in range(len(train_data_list))])
    ## todo: shuffer the whole list.
    return train_dataset,train_label
