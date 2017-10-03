# Work space directory
HOME_DIR = '/home/yuandy/guided_saliency/'
WEIGHTS_DIR = '/home/yuandy/guided_saliency/weights/'
#dir to save validation result
TEST_DIR = '/home/yuandy/guided_saliency/test/bigan_all_test22'

# Path to COCO raw data
pathToImages = '/home/yuandy/COCO_dataset/train_images/images'
#pathToMaps = '/home/yuandy/COCO_dataset/coco/PythonAPI/128_32_2down_COCO_guided'
pathToMaps = '/home/yuandy/saliency-salgan-2017/128_32_2down_COCO_prediction'
pathToTestMaps = '/home/yuandy/COCO_dataset/guided_saliency_gaussian_5_test_new'

# Path to geometry toy data
#pathToImages = '/home/yuandy/saliency-salgan-2017/geo_train'
#pathToMaps = '/home/yuandy/saliency-salgan-2017/geometry_sal_nocenter_new2'
#pathToMaps = '/home/yuandy/saliency-salgan-2017/geo_train_sal'
#pathToTestMaps = '/home/yuandy/saliency-salgan-2017/geo_test_sal_center_new2'

# Path to processed data
pathOutputImages = '/home/yuandy/COCO_dataset/processed_data/images256x192'
pathOutputMaps = '/home/yuandy/COCO_dataset/processed_data/saliency256x192'
pathOutputTestImages = '/home/yuandy/COCO_dataset/processed_data/imagesTest256x192'
pathOutputTestMaps = '/home/yuandy/COCO_dataset/processed_data/saliencyTest256x192'
'''
pathOutputImages = '/home/yuandy/COCO_dataset/processed_data/toy_images256x192'
pathOutputMaps = '/home/yuandy/COCO_dataset/processed_data/toy_saliency256x192'
pathOutputTestImages = '/home/yuandy/COCO_dataset/processed_data/toy_imagesTest256x192'
pathOutputTestMaps = '/home/yuandy/COCO_dataset/processed_data/toy_saliencyTest256x192'
'''
pathToPickle = '/home/yuandy/COCO_dataset/processed_data/128x128'

# Path to pickles which contains processed data
TRAIN_DATA_DIR = '/home/yuandy/COCO_dataset/processed_data/256x192/trainData.pickle'
VAL_DATA_DIR = '/home/yuandy/COCO_dataset/256x192/fix_validationData.pickle'
TEST_DATA_DIR = '/home/yuandy/COCO_dataset/256x192/testData.pickle'

# Path to vgg16 pre-trained weights
PATH_TO_VGG16_WEIGHTS = '/home/yuandy/guided_saliency/scripts/vgg16.pkl'

# Input image and saliency map size
# (WIDTH, HEIGHT)
#INPUT_SIZE = (256, 192)
INPUT_SIZE = (128, 128)
LOSS_TO_SAVE = './loss_bigan_all_test22.npz'
#LOSS_TO_SAVE = './loss_bigan_test6_norm.npz'
# Directory to keep snapshots
#DIR_TO_SAVE = 'home/yuandy/guided_saliency/weights_0.00001_biGAN'
DIR_TO_SAVE = 'home/yuandy/guided_saliency/weights/weights_bigan_all_test22'
#DIR_TO_SAVE = 'media/yuandy/weights/weights_bigan_test6_norm'
