import numpy as np
import tensorflow as tf

def ConMap2Mask_prob(c_map,hori_translation,verti_translation):
    '''
    continuous bilateral voting
    '''
    # if len(c_map.shape) == 3:
    #     c_map = tf.expand_dims(c_map, 0)   
    
    # print("cmap : ", c_map.shape)
    _, row, column, channel = c_map.shape
    # c_map = tf.squeeze(c_map)
    vote_out = tf.zeros(shape = [row, column, channel])

    # print(c_map[1,4].shape)
    right        = tf.matmul(c_map[ :, :, :, 4],hori_translation)
    left         = tf.matmul (c_map[:, :, :, 3],hori_translation.transpose(1,0))
    left_bottom  = tf.matmul(verti_translation.transpose(1,0), c_map[:, :,:, 5])
    left_bottom  = tf.matmul(left_bottom,hori_translation.transpose(1,0))
    right_above  = tf.matmul(verti_translation, c_map[:,:,:, 2])
    right_above  = tf.matmul(right_above,hori_translation)
    left_above   = tf.matmul(verti_translation, c_map[:,:,:, 0])
    left_above   = tf.matmul(left_above,hori_translation.transpose(1,0))
    bottom       = tf.matmul(verti_translation.transpose(1,0), c_map[:,:,:, 6])
    up           = tf.matmul(verti_translation, c_map[:,:,:, 1])
    right_bottom = tf.matmul(verti_translation.transpose(1,0), c_map[:,:,:, 7])
    right_bottom = tf.matmul(right_bottom,hori_translation)
    
    a1 = (c_map[:,:,:, 3]) * (right)        
    a2 = (c_map[:,:,:, 4]) * (left)
    a3 = (c_map[:,:,:, 1]) * (bottom)
    a4 = (c_map[:,:,:, 6]) * (up)
    a5 = (c_map[:,:,:, 2]) * (left_bottom)
    a6 = (c_map[:,:,:, 5]) * (right_above)
    a7 = (c_map[:,:,:, 0]) * (right_bottom)
    a8 = (c_map[:,:,:, 7]) * (left_above)
    
    vote_out = tf.stack([a7, a3, a5, a1, a2, a6, a4, a8], axis = -1)
    
    pred_mask = tf.math.reduce_mean(vote_out, axis=-1)
    pred_mask = tf.squeeze(pred_mask)
    return pred_mask        
            
        
def bv_test(y_pred):
    '''
    generate the continous global map from output connectivity map as final saliency output 
    via bilateral voting
    '''        
    #construct the translation matrix
    hori_translation = np.zeros(shape  = (y_pred.shape[2],y_pred.shape[2]), dtype = np.float32)
    for i in range(y_pred.shape[2]-1):
        hori_translation[i,i+1] = 1
    verti_translation = np.zeros(shape = (y_pred.shape[1],y_pred.shape[1]), dtype = np.float32)
    for j in range(y_pred.shape[1]-1):
        verti_translation[j,j+1] = 1    

    pred = ConMap2Mask_prob(y_pred,hori_translation,verti_translation)
    return pred            
            

def create_map(args, predictions):
    predictions = np.array(predictions[:, :, :, 0:args.num_classes * 8] )        
    num_samples, W, H, C = predictions.shape
    pred_data = np.zeros(shape = (num_samples, 256, 256, args.num_classes))
    for i in range(num_samples):
        sample = predictions[i]
        sample = np.expand_dims(sample, axis = 0)
        for x in range(args.num_classes):
            output_bicon = bv_test(sample[:, :, :, x * 8:(x + 1) * 8])  
            pred_data[i, :, :, x] = np.expand_dims(output_bicon, axis = 0)    
    predictions = pred_data
    predictions = np.squeeze(predictions)        
    predictions = np.argmax(predictions, axis = -1)  

    return predictions
      
      
  