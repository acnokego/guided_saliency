import os
import numpy as np
from tqdm import tqdm
import cv2
import glob
import theano
from utils import *
from constants import *
from models.autoencoder import autoencoder





def test(path_to_images, path_to_saliency, path_output_imgs, model_to_test=None):
    list_img_files = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(path_to_saliency, '*'))]
    # Load Data
    list_img_files.sort()
    if not os.path.exists(path_output_imgs):
        os.mkdir(path_output_imgs)
    for curr_file in tqdm(list_img_files, ncols=20):
        #print os.path.join(path_to_images, curr_file + '.jpg')
        
        img = cv2.cvtColor(cv2.imread(os.path.join(path_to_images, curr_file + '.jpg'), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        #img = np.random.random((128,128,3))*255
        sal = cv2.imread(os.path.join(path_to_saliency, curr_file + '.jpg'), cv2.IMREAD_GRAYSCALE)
        predict(model=model_to_test, image_stimuli=img, saliency_stimuli=sal, name=curr_file, path_output_imgs=path_output_imgs)


def main():
    # Create network
    model = autoencoder(INPUT_SIZE[0], INPUT_SIZE[1], batch_size=8)
    # Here need to specify the epoch of model sanpshot
    #load_weights(model.net, path='scripts/gen_', epochtoload=90)
    #load_weights_test(model.net['output_encoder'], path='weights_toy_center_0.0001_1layer/auto_', epochtoload=99)
    #load_weights_test(model.net['output_encoder'], path='fix_3layers_weights/auto_finetune_', epochtoload=60)
    # Here need to specify the path to images and output path
    #load_weights_test(model.net['output_encoder'], path='weights_resize_com_9_pool2_gs/auto_', epochtoload=27)
    #load_weights_test(model.encoder['output_encoder'], path='weights_auto_new_-9/auto_', epochtoload=27)
    #load_weights_test(model.encoder['output_encoder'], path='weights_bigan_noise_test11/biGAN_G_', epochtoload=20000)
    #load_weights_test(model.encoder['output_encoder'], path='weights_content_new/auto_', epochtoload=21)
    load_weights_test(model.encoder['output_encoder'], path='weights_bigan_all_test15/biGAN_G_', epochtoload=12000)
    #load_weights_test(model.net['output_encoder'], path='weights_3layers_small_resized/auto_', epochtoload=99)
    #test(path_to_images='../../COCO_dataset/train_images/images', path_to_saliency='../../COCO_dataset/temp_test_pool2_gs', path_output_imgs='../test', model_to_test=model)
    test(path_to_images='/media/yuandy/COCO_dataset/train_images/images', path_to_saliency='/media/yuandy/COCO_dataset/temp_test_max1.5', path_output_imgs='../test', model_to_test=model)

    #test_batch(path_to_images='/media/yuandy/COCO_dataset/train_images/images', path_to_saliency='/media/yuandy/COCO_dataset/temp_test_max1.5',name='test', model=model)
    
    #test(path_to_images='../../COCO_dataset/temp_test_two', path_to_saliency='../../COCO_dataset/temp_test_two_pool2_gs/', path_output_imgs='../test', model_to_test=model)
    #test(path_to_images='../../saliency-salgan-2017/geometry_temp_test', path_to_saliency='../../saliency-salgan-2017/geometry_temp_test_new2_center_sal', path_output_imgs='../test', model_to_test=model)

if __name__ == "__main__":
    main()
