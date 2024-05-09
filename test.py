import os
import cv2
import yaml
import tqdm
import glob
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


from model.network      import RetiFluidNet
from src.inference_utils import create_map

def set_parameters(params:dict) -> dict:
    """_summary_

    Args:
        param (dict): config dictionary parameters

    Returns:
        dict: compelete config dictionary parameters
    """
  
    assert os.path.isdir(params["path"]), "Dataset directory is not valid"
    os.makedirs(params["save_path"], exist_ok = True)
    data = {
        "path"                      : params["path"],
        "image_size"                : params["image_size"],
        "in_channels"               : params["in_channels"],
        "num_classes"               : params["num_classes"],
        "save_path"                 : params["save_path"],
        "checkpoint"                : params["checkpoint"],
        }
    return data

def inference(args):
    #Load Model and Results
    model = RetiFluidNet(args.num_classes, (args.image_size,args.image_size,args.in_channels))()
    model.load_weights(args.checkpoint)
    
    data_path = glob.glob(args.path + "/*")
    
    for image_path in tqdm.tqdm(data_path):
        name = image_path.split(os.path.sep)[-1].split(".")[0]
        image = cv2.imread(image_path, 0)
        image = cv2.resize(image / 255.0, (256,256))
        image = np.expand_dims(image, axis = 0)
        output = model.predict(image)
        prediction = create_map(args, output) * 255.0
        cv2.imwrite("{}/{}.png".format(args.save_path, name), prediction)
    

if __name__ == '__main__':
    params = yaml.safe_load(open("params.yaml"))["inference"]
    params = set_parameters(params)
    args = argparse.Namespace(**params)
    predictions = inference(args)