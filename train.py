
# In[]
import os
import yaml
import argparse
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from src            import train_utils
from src.DataReader import DataReader
from model.network  import RetiFluidNet
from src.losses     import Losses
from src.results    import Results

# In[]


def set_parameters(params:dict) -> dict:
    """_summary_

    Args:
        param (dict): config dictionary parameters

    Returns:
        dict: compelete config dictionary parameters
    """
  
    assert os.path.isdir(params["path"]), "Dataset directory is not valid"
    assert params["num_classes"] == len(params["classes_names"]), "There must be a name for each class"
  
    data = {
        "dataset_name"              : params["dataset_name"],
        "path"                      : params["path"],
        "image_size"                : params["image_size"],
        "in_channels"               : params["in_channels"],
        "num_classes"               : params["num_classes"],
        "seed"                      : params["seed"],
        "epochs"                    : params["epochs"],
        "batch_size"                : params["batch_size"],
        "buffer_size"               : params["buffer_size"],        
        "split"                     : params["split"],
        "initial_learning_rate"     : params["initial_learning_rate"],
        "image_format"              : params["image_format"],
        "mask_format"               : params["mask_format"],
        "classes_names"             : params["classes_names"],
        "autotune"                  : tf.data.experimental.AUTOTUNE,
        }
    return data


def train(args):
    
    data_reader  = DataReader(args) #Define Data Reader Object
    loss         = Losses(args)     #Define Loss Object
    results      = Results(args)    #Define Result Object
    model        = RetiFluidNet(args.num_classes,  #Define Model
                                (args.image_size,    
                                 args.image_size,
                                 args.in_channels)
                                )()


    
    data_path = train_utils.get_casses(args.path) #Get Training Casses

    train_path = data_path[:int(args.split * len(data_path))]
    val_path   = data_path[int(args.split * len(data_path)):]
    
    train_path, val_path = data_reader.get_trainPath_and_valPath(train_path, val_path) 
    train_data, val_data = data_reader.get_data_for_train(train_path, val_path)
        
    
    train_data = train_data.shuffle(buffer_size = args.buffer_size, seed=args.seed).batch(args.batch_size).prefetch(buffer_size = args.autotune)
    val_data   = val_data.batch(1).prefetch(buffer_size = args.autotune)


    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(train_utils.decay_schedule)


    if not os.path.exists(args.dataset_name):
        os.mkdir(args.dataset_name)

    # Creating Callbacks
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(args.dataset_name+"/model_checkpoint.hdf5",save_best_only=True) 
                                                        

    model.compile(optimizer = tf.keras.optimizers.RMSprop(args.initial_learning_rate), 
                    loss = loss.training_loss)

    
    history = model.fit(train_data,
                        epochs=args.epochs,
                        callbacks=[lr_scheduler])
    model.save(args.dataset_name+"/model_epoch%s.hdf5"%(args.epochs))
    
    with open(args.dataset_name+"/model_history.npy", 'wb') as f:
        np.save(f, history.history)
                                                                             

    predictions = []
    for image, _ in tqdm(val_data):  
        temp = model.predict(image)[:, :, :, 0:args.num_classes * 8]
        predictions.append(temp)
    acc_mean, dice_mean, f1_score_mean, precision_mean, bacc_mean, recall_mean, iou_mean = results.results_per_layer(predictions, val_data)


if __name__ == '__main__':
    params = yaml.safe_load(open("params.yaml"))["train"]
    params = set_parameters(params)
    args = argparse.Namespace(**params)
    predictions = train(args)
    




