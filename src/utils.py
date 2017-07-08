import tensorflow as tf
import os
import numpy as np
import zipfile

CAT = 0
DOG = 1

cwd = os.getcwd()

def prepare_file():
  file_list = ['train', 'test']
  valid = True

  for i in range(len(file_list)):
    filename = file_list[i] + '.zip'
    dest_filename = os.path.join(cwd, 'data', filename)

    if not os.path.exists(dest_filename):
      print('Please download ' + filename + ' and put on src/data folder')
      url = "https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/download/"
      print(url + filename)
      valid = False
      continue
    
    images_path = os.path.join(cwd, 'data', filename)

    zip = zipfile.ZipFile(dest_filename)
    if not os.path.exists(images_path):
        print('Extracting...')
        zip.extractall(os.path.join(cwd, 'data'))
      
  return valid

def read_image_label_list(folder_dir):
    dir_list = os.listdir(os.path.join(cwd, folder_dir))
    
    filenames = []
    labels = []
    
    for i, d in enumerate(dir_list):
        if folder_dir == 'train':
            if re.search("cat", d):
                labels.append(CAT)
            else:
                labels.append(DOG)
        else:
            labels.append(-1)
        filenames.append(d)
    
    return filenames, labels

def read_images_from_disk(input_queue):
    filename = input_queue[0]
    label = input_queue[1]
    
    file_contents = tf.read_file(filename)
    image = tf.image.decode_image(file_contents, channels=3)
    image.set_shape([None, None, 3])
    
    return image, label

def read_img(data_dir, batch_size, shuffle):
    def input_fn():
        image_list, label_list = read_image_label_list(data_dir)
        
        images = tf.convert_to_tensor(image_list, dtype=tf.string)
        labels = tf.convert_to_tensor(label_list, dtype=tf.int32)
        
        input_queue = tf.train.slice_input_producer(
            [images, labels],
            shuffle=shuffle,
            capacity=batch_size * 2,
            name="file_input_queue"
        )
        
        image, label = read_images_from_disk(input_queue)
        
        image = tf.image.resize_images(image, (256, 256), tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        
        image_batch, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            capacity=batch_size * 2,
            num_threads=1,
            name="batch_queue",
            allow_smaller_final_batch=False
        )
        
        return tf.identity(image_batch, name="features"), tf.identity(label_batch, name="label")
    
    return input_fn
            