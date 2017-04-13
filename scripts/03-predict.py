import os
import numpy as np
from tqdm import tqdm
import cv2
import glob
from utils import *
from constants import *
from models.autoencoder import autoencoder


def test(path_to_images, path_to_saliency, path_output_imgs, model_to_test=None):
    list_img_files = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(path_to_saliency, '*'))]
    # Load Data
    list_img_files.sort()
    for curr_file in tqdm(list_img_files, ncols=20):
        print os.path.join(path_to_images, curr_file + '.jpg')
        img = cv2.cvtColor(cv2.imread(os.path.join(path_to_images, curr_file + '.jpg'), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        sal = cv2.imread(os.path.join(path_to_saliency, curr_file + '.jpg'), cv2.IMREAD_GRAYSCALE)
        predict(model=model_to_test, image_stimuli=img, saliency_stimuli=sal, name=curr_file, path_output_imgs=path_output_imgs)


def main():
    # Create network
    model = autoencoder(INPUT_SIZE[0], INPUT_SIZE[1], batch_size=8)
    # Here need to specify the epoch of model sanpshot
    load_weights_test(model.net['output_encoder'], path='test_finetune_weights/auto_', epochtoload=9)
    # Here need to specify the path to images and output path
    test(path_to_images='../../COCO_dataset/train_images/images', path_to_saliency='../../COCO_dataset/temp_test/sal', path_output_imgs='../test', model_to_test=model)

if __name__ == "__main__":
    main()
