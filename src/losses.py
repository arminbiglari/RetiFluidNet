import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

class Losses:
    
    def __init__(self, args):
        self.loss =  tf.keras.losses.BinaryCrossentropy()
        self.args = args
    
    @tf.function   
    def dice_coeff(self, y_true, y_pred):
        smooth = 1.
        # Flatten
        y_true_f = tf.reshape(tf.keras.backend.one_hot(tf.cast(y_true, 'int32'),num_classes = self.args.num_classes), [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
        return score
    
    @tf.function
    def dice_coeff_bicon(self, y_true, y_pred):
        smooth = 1.
        # Flatten
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
        return score
    
    
    @tf.function    
    def dice_loss(self, y_true, y_pred):
        loss = 1 - self.dice_coeff(y_true, y_pred)
        return loss
    
    @tf.function    
    def dice_loss_bicon(self, y_true, y_pred):
        loss = 1 - self.dice_coeff_bicon(y_true, y_pred)
        return loss
    

    def gen_dice(self, y_true, y_pred, eps=1e-6):
        """both tensors are [b, h, w, classes] and y_pred is in logit form"""

        y_true = tf.keras.backend.one_hot(tf.cast(y_true, 'int32'),num_classes = self.args.num_classes)
        y_true_shape = tf.shape(y_true)

        y_true = tf.reshape(y_true, [-1, y_true_shape[1]*y_true_shape[2], y_true_shape[3]])
        y_pred = tf.reshape(y_pred, [-1, y_true_shape[1]*y_true_shape[2], y_true_shape[3]])


        counts = tf.reduce_sum(y_true, axis=1)
        weights = 1. / (counts ** 4)
        weights = tf.where(tf.math.is_finite(weights), weights, eps)

        multed = tf.reduce_sum(y_true * y_pred, axis=1)
        summed = tf.reduce_sum(y_true + y_pred, axis=1)

        numerators = tf.reduce_sum(weights*multed, axis=-1)
        denom = tf.reduce_sum(weights*summed, axis=-1)
        dices = 1. - 2. * numerators / denom
        dices = tf.where(tf.math.is_finite(dices), dices, tf.zeros_like(dices))
        return tf.reduce_mean(dices)
    
    @tf.function
    def edge_loss(self, glo_map,vote_out,edge,target):
        pred_mask_min  = tf.math.reduce_min(vote_out , axis = -1)
        pred_mask_min  = 1 - pred_mask_min
        pred_mask_min  = pred_mask_min * edge
        decouple_map   = glo_map*(1-edge)+pred_mask_min
        minloss = self.loss(target, decouple_map)
        return minloss
               
    @tf.function          
    def Bilater_voting(self, c_map,hori_translation,verti_translation):
        
        _, row, column, channel = c_map.shape
        
        right        = tf.matmul(c_map[ :, :, :, 4],hori_translation)
        left         = tf.matmul (c_map[:, :, :, 3],tf.transpose(hori_translation, perm = [1, 0]))
        left_bottom  = tf.matmul(tf.transpose(verti_translation, perm = [1, 0]), c_map[:, :,:, 5]) 
        left_bottom  = tf.matmul(left_bottom,tf.transpose(hori_translation, perm = [1, 0]))
        right_above  = tf.matmul(verti_translation, c_map[:,:,:, 2])
        right_above  = tf.matmul(right_above,hori_translation)
        left_above   = tf.matmul(verti_translation, c_map[:,:,:, 0])
        left_above   = tf.matmul(left_above,tf.transpose(hori_translation, perm = [1, 0]))
        bottom       = tf.matmul(tf.transpose(verti_translation, perm = [1, 0]), c_map[:,:,:, 6])
        up           = tf.matmul(verti_translation, c_map[:,:,:, 1])
        right_bottom = tf.matmul(tf.transpose(verti_translation, perm = [1, 0]), c_map[:,:,:, 7])
        right_bottom = tf.matmul(right_bottom,hori_translation)
        
        a1 = tf.multiply(c_map[:,:,:, 3], right)       
        a2 = tf.multiply(c_map[:,:,:, 4], left)
        a3 = tf.multiply(c_map[:,:,:, 1], bottom)
        a4 = tf.multiply(c_map[:,:,:, 6], up)
        a5 = tf.multiply(c_map[:,:,:, 2], left_bottom)
        a6 = tf.multiply(c_map[:,:,:, 5], right_above)
        a7 = tf.multiply(c_map[:,:,:, 0], right_bottom)
        a8 = tf.multiply(c_map[:,:,:, 7], left_above)
        
        vote_out = tf.stack([a7, a3, a5, a1, a2, a6, a4, a8], axis = -1)
        return vote_out  
    
    def sal2conn(self, mask):
        mask        = np.squeeze(np.array(mask))        
        if len(mask.shape) == 2:
            mask = tf.expand_dims(mask, 0)
            
        batch, rows, cols  = mask.shape
        conn        = np.zeros(shape = (batch, rows, cols, 8))
        up          = np.zeros(shape = (batch, rows, cols))
        down        = np.zeros(shape = (batch, rows, cols))
        left        = np.zeros(shape = (batch, rows, cols))
        right       = np.zeros(shape = (batch, rows, cols))
        up_left     = np.zeros(shape = (batch, rows, cols))
        up_right    = np.zeros(shape = (batch, rows, cols))
        down_left   = np.zeros(shape = (batch, rows, cols))
        down_right  = np.zeros(shape = (batch, rows, cols))
    
        up[:,:rows-1, :]             = mask[:,1:rows,:]
        down[:,1:rows,:]             = mask[:,0:rows-1,:]
        left[:,:,:cols-1]            = mask[:,:,1:cols]
        right[:,:,1:cols]            = mask[:,:,:cols-1]
        up_left[:,0:rows-1,0:cols-1] = mask[:,1:rows,1:cols]
        up_right[:,0:rows-1,1:cols]  = mask[:,1:rows,0:cols-1]
        down_left[:,1:rows,0:cols-1] = mask[:,0:rows-1,1:cols]
        down_right[:,1:rows,1:cols]  = mask[:,0:rows-1,0:cols-1]  
        
        conn[:,:,:,0] = mask*down_right
        conn[:,:,:,1] = mask*down
        conn[:,:,:,2] = mask*down_left
        conn[:,:,:,3] = mask*right
        conn[:,:,:,4] = mask*left
        conn[:,:,:,5] = mask*up_right
        conn[:,:,:,6] = mask*up
        conn[:,:,:,7] = mask*up_left
        conn = conn.astype(np.float32)
        
        return conn
    
    @tf.function
    def tf_sal2conn(self, mask):
        output = tf.numpy_function(self.sal2conn, [mask], tf.float32) 
        return output 

    def numpy_full_like(self, x, y):
        return np.full_like(x, y)
    
    @tf.function
    def tf_full_like(self, x, y):
        return tf.numpy_function(self.numpy_full_like, [x, y], tf.float32)          

                
    @tf.function
    def bicon_loss_calculator(self, y_ture, y_pred):

        y_ture8 = self.tf_sal2conn(y_ture) 
        sum_conn = tf.math.reduce_sum(y_pred, axis = -1)
        edge  = tf.where(sum_conn < 8, self.tf_full_like(sum_conn, 1), self.tf_full_like(sum_conn, 0))
        edge1 = tf.where(sum_conn > 0, self.tf_full_like(sum_conn, 1), self.tf_full_like(sum_conn, 0))           
        edge  = tf.multiply(edge, edge1)
        hori_translation = np.zeros(shape  = (y_pred.shape[2],y_pred.shape[2]), dtype = np.float32)
        for i in range(y_pred.shape[2]-1):
            hori_translation[i,i+1] = 1

        verti_translation = np.zeros(shape = (y_pred.shape[1],y_pred.shape[1]), dtype = np.float32)
        for j in range(y_pred.shape[1]-1):
            verti_translation[j,j+1] = 1

        vote_out = self.Bilater_voting(y_pred,hori_translation,verti_translation) 


        glo_map = tf.math.reduce_max(vote_out, axis = -1) 


        de_loss  = self.edge_loss(glo_map,vote_out,edge,y_ture)   

        dice_loss = self.dice_loss_bicon(y_ture, glo_map)


        loss_con_const = self.loss(y_ture8, y_pred)

        loss = 0.5 * (de_loss) + loss_con_const + dice_loss
 
        return loss

    @tf.function
    def training_loss(self, y_true, y_pred):
       
        y_true_bicon = tf.keras.backend.one_hot(tf.cast(y_true, 'int32'),num_classes = self.args.num_classes)

        y_true_bicon = tf.squeeze(y_true_bicon) 
        
        
        dice_layer_index = 5 * self.args.num_classes * 8
        outputs = y_pred[:, :, :, dice_layer_index + self.args.num_classes * (1 - 1):dice_layer_index + self.args.num_classes * 1] #Side output of the main output
        output4 = y_pred[:, :, :, dice_layer_index + self.args.num_classes * (2 - 1):dice_layer_index + self.args.num_classes * 2] #Side output4
        output3 = y_pred[:, :, :, dice_layer_index + self.args.num_classes * (3 - 1):dice_layer_index + self.args.num_classes * 3] #Side output3
        output2 = y_pred[:, :, :, dice_layer_index + self.args.num_classes * (4 - 1):dice_layer_index + self.args.num_classes * 4] #Side output2
        output1 = y_pred[:, :, :, dice_layer_index + self.args.num_classes * (5 - 1):dice_layer_index + self.args.num_classes * 5] #Side output1     

        bicon_loss = {}
        for i in range(5):
            number_of_out_channels = self.args.num_classes * 8
            bicon_output   = y_pred[:, :, :, i * number_of_out_channels : (i + 1) * number_of_out_channels]
            bicon_loss["decoder_" + str(i)] = 0 
            for x in range(self.args.num_classes):
                output_bicon = bicon_output[:, :, :, x * 8:(x + 1) * 8]
                loss_bicon   = self.bicon_loss_calculator(y_true_bicon[:, :, :, x], output_bicon)
                bicon_loss["decoder_" + str(i)] += loss_bicon

         
        Multi_scale_bicon_loss = (1/8.0) * bicon_loss["decoder_4"] + (1/4.0) * bicon_loss["decoder_3"] + (1/2.0) * bicon_loss["decoder_2"] + bicon_loss["decoder_1"] + bicon_loss["decoder_0"]
        
        loss_side_main = self.dice_loss(y_true, outputs) #Main output
        loss_side_1    = self.dice_loss(y_true, output1) #output of the encoder 1
        loss_side_2    = self.dice_loss(y_true, output2) #output of the encoder 2
        loss_side_3    = self.dice_loss(y_true, output3) #output of the encoder 3
        loss_side_4    = self.dice_loss(y_true, output4) #output of the encoder 4
        
        Multi_scale_dice_loss  = 1 * loss_side_main + 1 * loss_side_1 + (1/2.0) * loss_side_2 + (1/4.0) * loss_side_3 + (1/8.0) * loss_side_4
        joint_loss = 0.05 * (Multi_scale_bicon_loss) + 1.0 * Multi_scale_dice_loss
        # joint_loss = Multi_scale_bicon_loss
        return joint_loss

    
    @tf.function           
    def ConMap2Mask_prob(self, c_map,hori_translation,verti_translation):
        '''
        continuous bilateral voting
        '''

        _, row, column, channel = c_map.shape
        vote_out = tf.zeros(shape = [row, column, channel])
    
        right        = tf.matmul(c_map[ :, :, :, 4],hori_translation)
        left         = tf.matmul (c_map[:, :, :, 3],tf.transpose(hori_translation, perm = [1, 0]))
        left_bottom  = tf.matmul(tf.transpose(verti_translation, perm = [1, 0]), c_map[:, :,:, 5]) 
        left_bottom  = tf.matmul(left_bottom,tf.transpose(hori_translation, perm = [1, 0]))
        right_above  = tf.matmul(verti_translation, c_map[:,:,:, 2])
        right_above  = tf.matmul(right_above,hori_translation)
        left_above   = tf.matmul(verti_translation, c_map[:,:,:, 0])
        left_above   = tf.matmul(left_above,tf.transpose(hori_translation, perm = [1, 0]))
        bottom       = tf.matmul(tf.transpose(verti_translation, perm = [1, 0]), c_map[:,:,:, 6])
        up           = tf.matmul(verti_translation, c_map[:,:,:, 1])
        right_bottom = tf.matmul(tf.transpose(verti_translation, perm = [1, 0]), c_map[:,:,:, 7])
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
            
    @tf.function 
    def ConMap2Mask_prob_new(self, c_map,hori_translation,verti_translation):
        '''
        continuous bilateral voting
        '''
        _, row, column, channel = c_map.shape
        vote_out = tf.zeros(shape = [row, column, channel])
    
        right        = tf.matmul(c_map[ :, :, :, 4],hori_translation)
        left         = tf.matmul (c_map[:, :, :, 3],tf.transpose(hori_translation, perm = [1, 0]))
        left_bottom  = tf.matmul(tf.transpose(verti_translation, perm = [1, 0]), c_map[:, :,:, 5]) 
        left_bottom  = tf.matmul(left_bottom,tf.transpose(hori_translation, perm = [1, 0]))
        right_above  = tf.matmul(verti_translation, c_map[:,:,:, 2])
        right_above  = tf.matmul(right_above,hori_translation)
        left_above   = tf.matmul(verti_translation, c_map[:,:,:, 0])
        left_above   = tf.matmul(left_above,tf.transpose(hori_translation, perm = [1, 0]))
        bottom       = tf.matmul(tf.transpose(verti_translation, perm = [1, 0]), c_map[:,:,:, 6])
        up           = tf.matmul(verti_translation, c_map[:,:,:, 1])
        right_bottom = tf.matmul(tf.transpose(verti_translation, perm = [1, 0]), c_map[:,:,:, 7])
        right_bottom = tf.matmul(right_bottom,hori_translation)
        
        
        # print(a1[0][0][100])
        a1 = (c_map[:,:,:, 3]) * (right)        
        a2 = (c_map[:,:,:, 4]) * (left)
        a3 = (c_map[:,:,:, 1]) * (bottom)
        a4 = (c_map[:,:,:, 6]) * (up)
        a5 = (c_map[:,:,:, 2]) * (left_bottom)
        a6 = (c_map[:,:,:, 5]) * (right_above)
        a7 = (c_map[:,:,:, 0]) * (right_bottom)
        a8 = (c_map[:,:,:, 7]) * (left_above)
        
        vote_out = tf.stack([a7, a3, a5, a1, a2, a6, a4, a8], axis = -1)
        
        pred_mask = tf.math.reduce_max(vote_out, axis=-1) 
        pred_mask = tf.squeeze(pred_mask)
        return pred_mask        
            
    @tf.function         
    def bv_test(self, y_pred):
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
    
        pred = self.ConMap2Mask_prob(y_pred,hori_translation,verti_translation)
        return pred            
    
    @tf.function             
    def bv_test_new(self, y_pred):
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
    
        pred = self.ConMap2Mask_prob_new(y_pred,hori_translation,verti_translation)
        return pred    
                    
