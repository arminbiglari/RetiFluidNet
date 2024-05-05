import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import glob
import tensorflow_addons as tfa
import random

  

class DataReader:
  
    def __init__(self, args):
      self.args = args
      
    def load_image(self, path):    
      #Read image
      image = tf.io.read_file(path)
      image = tf.image.decode_jpeg(image)   
      image = tf.cast(image, tf.float32) / 255.0      
      image = tf.image.resize(image, [self.args.image_size, self.args.image_size])
     
      #Read mask
      mask_path = tf.strings.regex_replace(path, self.args.image_format, self.args.mask_format)
      mask = tf.io.read_file(mask_path)
      mask = tf.image.decode_png(mask)
      mask = tf.cast(mask, tf.float32) / 255.0 
      mask = tf.image.resize(mask, [self.args.image_size, self.args.image_size], method = 'nearest')
      

      return image, mask


    def load_image_rotate_5(self, path):
    
      angle = 0.05
      #Read image
      image = tf.io.read_file(path)
      image = tf.image.decode_jpeg(image)
      image = tf.cast(image, tf.float32) / 255.0
      image = tf.image.resize(image, [self.args.image_size, self.args.image_size])
      image = tfa.image.rotate(image, angle)
    
      #Read mask
      mask_path = tf.strings.regex_replace(path, self.args.image_format, self.args.mask_format)      
      mask = tf.io.read_file(mask_path)
      mask = tf.image.decode_png(mask)
      mask = tf.cast(mask, tf.float32) / 255.0 
      mask = tf.image.resize(mask, [self.args.image_size, self.args.image_size], method = 'nearest')
      mask = tfa.image.rotate(mask, angle)
       
    
      return image, mask

    def load_image_rotate_5_r(self, path):
    
      angle = -0.05
      #Read image
      image = tf.io.read_file(path)
      image = tf.image.decode_jpeg(image) 
      image = tf.cast(image, tf.float32) / 255.0
      image = tf.image.resize(image, [self.args.image_size, self.args.image_size])
      image = tfa.image.rotate(image, angle)
     
      #Read mask
      mask_path = tf.strings.regex_replace(path, self.args.image_format, self.args.mask_format)      
      mask = tf.io.read_file(mask_path)
      mask = tf.image.decode_png(mask)
      mask = tf.cast(mask, tf.float32) / 255.0 
      mask = tf.image.resize(mask, [self.args.image_size, self.args.image_size], method = 'nearest')
      mask = tfa.image.rotate(mask, angle)
       
    
      return image, mask


    def load_image_rotate_2(self, path):  
        
      angle = 0.02
      #Read image
      image = tf.io.read_file(path)
      image = tf.image.decode_jpeg(image)
      image = tf.cast(image, tf.float32) / 255.0
      image = tf.image.resize(image, [self.args.image_size, self.args.image_size])
      image = tfa.image.rotate(image, angle)
    
      #Read mask
      mask_path = tf.strings.regex_replace(path, self.args.image_format, self.args.mask_format)      
      mask = tf.io.read_file(mask_path)
      mask = tf.image.decode_png(mask)
      mask = tf.cast(mask, tf.float32) / 255.0 
      mask = tf.image.resize(mask, [self.args.image_size, self.args.image_size], method = 'nearest')
      mask = tfa.image.rotate(mask, angle)
       
    
      return image, mask

    def load_image_rotate_2_r(self, path):
    
      angle = -0.02
      #Read image
      image = tf.io.read_file(path)
      image = tf.image.decode_jpeg(image)
      image = tf.cast(image, tf.float32) / 255.0
      image = tf.image.resize(image, [self.args.image_size, self.args.image_size])
      image = tfa.image.rotate(image, angle)
    
      #Read mask
      mask_path = tf.strings.regex_replace(path, self.args.image_format, self.args.mask_format)      
      mask = tf.io.read_file(mask_path)
      mask = tf.image.decode_png(mask)
      mask = tf.cast(mask, tf.float32) / 255.0 
      mask = tf.image.resize(mask, [self.args.image_size, self.args.image_size], method = 'nearest')
      mask = tfa.image.rotate(mask, angle)
       
    
      return image, mask

    def load_image_rotate_1(self, path):
    
      angle = 0.01
      #Read image
      image = tf.io.read_file(path)
      image = tf.image.decode_jpeg(image)
      image = tf.cast(image, tf.float32) / 255.0
      image = tf.image.resize(image, [self.args.image_size, self.args.image_size])
      image = tfa.image.rotate(image, angle)
    
      #Read mask
      mask_path = tf.strings.regex_replace(path, self.args.image_format, self.args.mask_format)      
      mask = tf.io.read_file(mask_path)
      mask = tf.image.decode_png(mask)
      mask = tf.cast(mask, tf.float32) / 255.0 
      mask = tf.image.resize(mask, [self.args.image_size, self.args.image_size], method = 'nearest')
      mask = tfa.image.rotate(mask, angle)
       
    
      return image, mask
    
    def load_image_rotate_1_r(self, path):
      
      angle = -0.01
      #Read image
      image = tf.io.read_file(path)
      image = tf.image.decode_jpeg(image)
      image = tf.cast(image, tf.float32) / 255.0
      image = tf.image.resize(image, [self.args.image_size, self.args.image_size])
      image = tfa.image.rotate(image, angle)
    
      #Read mask
      mask_path = tf.strings.regex_replace(path, self.args.image_format, self.args.mask_format)      
      mask = tf.io.read_file(mask_path)
      mask = tf.image.decode_png(mask)
      mask = tf.cast(mask, tf.float32) / 255.0 
      mask = tf.image.resize(mask, [self.args.image_size, self.args.image_size], method = 'nearest')
      mask = tfa.image.rotate(mask, angle)
       
    
      return image, mask

  
    
    def load_image_fliplr(self, path):
           
      #Read image
      image = tf.io.read_file(path)
      image = tf.image.decode_jpeg(image)
      image = tf.cast(image, tf.float32) / 255.0
      image = tf.image.resize(image, [self.args.image_size, self.args.image_size])
      image = tf.image.flip_left_right(image)  
    
      #Read mask
      mask_path = tf.strings.regex_replace(path, self.args.image_format, self.args.mask_format)      
      mask = tf.io.read_file(mask_path)
      mask = tf.image.decode_png(mask)
      mask = tf.cast(mask, tf.float32) / 255.0 
      mask = tf.image.resize(mask, [self.args.image_size, self.args.image_size], method = 'nearest')
      mask = tf.image.flip_left_right(mask)
             
    
      return image, mask


    def load_image_translate_1_1(self, path):
    
      translation = random.randint(-20, 20)
      a = 1
      b = 1
      #Read image
      image = tf.io.read_file(path)
      image = tf.image.decode_jpeg(image)
      image = tf.cast(image, tf.float32) / 255.0
      image = tf.image.resize(image, [self.args.image_size, self.args.image_size])
      image = tfa.image.translate(image, [a*translation, b*translation])
    
    
      #Read mask
      mask_path = tf.strings.regex_replace(path, self.args.image_format, self.args.mask_format)      
      mask = tf.io.read_file(mask_path)
      mask = tf.image.decode_png(mask)
      mask = tf.cast(mask, tf.float32) / 255.0 
      mask = tf.image.resize(mask, [self.args.image_size, self.args.image_size], method = 'nearest')
      mask = tfa.image.translate(mask, [a*translation, b*translation])
       
      
      return image, mask

    def load_image_translate_n_1_1(self, path):
    
      translation = random.randint(-20, 20)
      a = -1
      b = 1
      #Read image
      image = tf.io.read_file(path)
      image = tf.image.decode_jpeg(image)
      image = tf.cast(image, tf.float32) / 255.0
      image = tf.image.resize(image, [self.args.image_size, self.args.image_size])
      image = tfa.image.translate(image, [a*translation, b*translation])
    
      #Read mask
      mask_path = tf.strings.regex_replace(path, self.args.image_format, self.args.mask_format)      
      mask = tf.io.read_file(mask_path)
      mask = tf.image.decode_png(mask)
      mask = tf.cast(mask, tf.float32) / 255.0 
      mask = tf.image.resize(mask, [self.args.image_size, self.args.image_size], method = 'nearest')
      mask = tfa.image.translate(mask, [a*translation, b*translation])
       
      
      return image, mask

    def load_image_translate_1_n_1(self, path):
    
      translation = random.randint(-20, 20)
      a = 1
      b = -1
      #Read image
      image = tf.io.read_file(path)
      image = tf.image.decode_jpeg(image) 
      image = tf.cast(image, tf.float32) / 255.0
      image = tf.image.resize(image, [self.args.image_size, self.args.image_size])
      image = tfa.image.translate(image, [a*translation, b*translation])
    
      #Read mask
      mask_path = tf.strings.regex_replace(path, self.args.image_format, self.args.mask_format)      
      mask = tf.io.read_file(mask_path)
      mask = tf.image.decode_png(mask)
      mask = tf.cast(mask, tf.float32) / 255.0 
      mask = tf.image.resize(mask, [self.args.image_size, self.args.image_size], method = 'nearest')
      mask = tfa.image.translate(mask, [a*translation, b*translation])
       
      
      return image, mask

    def load_image_translate_n_1_n_1(self, path):
    
      translation = random.randint(-20, 20)
      a = -1
      b = -1
      #Read image
      image = tf.io.read_file(path)
      image = tf.image.decode_jpeg(image)
      image = tf.cast(image, tf.float32) / 255.0
      image = tf.image.resize(image, [self.args.image_size, self.args.image_size])
      image = tfa.image.translate(image, [a*translation, b*translation])
    
      #Read mask
      mask_path = tf.strings.regex_replace(path, self.args.image_format, self.args.mask_format)      
      mask = tf.io.read_file(mask_path)
      mask = tf.image.decode_png(mask)
      mask = tf.cast(mask, tf.float32) / 255.0 
      mask = tf.image.resize(mask, [self.args.image_size, self.args.image_size], method = 'nearest')
      mask = tfa.image.translate(mask, [a*translation, b*translation])
       
      
      return image, mask


    def load_image_contrast_n(self, path):
    
      contrast_factor = 0.5
      #Read image
      image = tf.io.read_file(path)
      image = tf.image.decode_jpeg(image)
      image = tf.image.adjust_contrast(image, contrast_factor)
      image = tf.cast(image, tf.float32) / 255.0
      image = tf.image.resize(image, [self.args.image_size, self.args.image_size])
    
      #Read mask
      mask_path = tf.strings.regex_replace(path, self.args.image_format, self.args.mask_format)      
      mask = tf.io.read_file(mask_path)
      mask = tf.image.decode_png(mask)
      mask = tf.cast(mask, tf.float32) / 255.0 
      mask = tf.image.resize(mask, [self.args.image_size, self.args.image_size], method = 'nearest')
       
      
      return image, mask
  
    def load_image_contrast_p(self, path):
    
      contrast_factor = 1.2
      #Read image
      image = tf.io.read_file(path)
      image = tf.image.decode_jpeg(image)
      image = tf.image.adjust_contrast(image, contrast_factor)
      image = tf.cast(image, tf.float32) / 255.0
      image = tf.image.resize(image, [self.args.image_size, self.args.image_size])
    
      #Read mask
      mask_path = tf.strings.regex_replace(path, self.args.image_format, self.args.mask_format)      
      mask = tf.io.read_file(mask_path)
      mask = tf.image.decode_png(mask)
      mask = tf.cast(mask, tf.float32) / 255.0 
      mask = tf.image.resize(mask, [self.args.image_size, self.args.image_size], method = 'nearest')
       
      
      return image, mask


    def get_data_for_train(self, train_path, val_path):
        
        train_data_orig = train_path.map(self.load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        val_data        = val_path.map(self.load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        train_data_con_p       = train_path.map(self.load_image_contrast_p, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_data_con_n       = train_path.map(self.load_image_contrast_n, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_data_rotate_1    = train_path.map(self.load_image_rotate_1, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_data_rotate_1_r  = train_path.map(self.load_image_rotate_1_r, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_data_rotate_2    = train_path.map(self.load_image_rotate_2, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_data_rotate_2_r  = train_path.map(self.load_image_rotate_2_r, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_data_rotate_5    = train_path.map(self.load_image_rotate_5, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_data_rotate_5_r  = train_path.map(self.load_image_rotate_5_r, num_parallel_calls=tf.data.experimental.AUTOTUNE)    
        train_data_fliplr      = train_path.map(self.load_image_fliplr, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_data_translate_1 = train_path.map(self.load_image_translate_1_1, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_data_translate_2 = train_path.map(self.load_image_translate_n_1_1, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_data_translate_3 = train_path.map(self.load_image_translate_1_n_1, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_data_translate_4 = train_path.map(self.load_image_translate_n_1_n_1, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        train_data = train_data_orig.concatenate(train_data_con_p).concatenate(train_data_fliplr).concatenate(train_data_con_n).concatenate(train_data_rotate_1).concatenate(train_data_rotate_1_r).concatenate(train_data_rotate_2).concatenate(train_data_rotate_2_r).concatenate(train_data_rotate_5).concatenate(train_data_rotate_5_r).concatenate(train_data_translate_1).concatenate(train_data_translate_2).concatenate(train_data_translate_3).concatenate(train_data_translate_4)
        
        print("Number of training samples are {}".format(len(train_data)))
        print("Number of validation samples are {}".format(len(val_data)))
        
        return train_data, val_data

    def get_trainPath_and_valPath(self, train_path, val_path):
                     
        train = []
        for path in train_path:
            for image_path in glob.glob(path + '/*'):
                if self.args.mask_format not in image_path and ".E2E" not in image_path:
                  train.append(image_path)
    
        validation = []
        for path in val_path:
            for image_path in glob.glob(path + '/*'):
                if self.args.mask_format not in image_path and ".E2E" not in image_path:
                  validation.append(image_path)
    
        train        = tf.data.Dataset.from_tensor_slices(train)            
        validation   = tf.data.Dataset.from_tensor_slices(validation)
        
        return train, validation
    



