import tensorflow as tf
import os
import numpy as np
import zipfile

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