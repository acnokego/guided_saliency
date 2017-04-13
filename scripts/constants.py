# Work space directory
HOME_DIR = '/home/yuandy/guided_saliency/'

# Path to SALICON raw data
pathToImages = '/home/yuandy/COCO_dataset/train_images/images'
pathToMaps = '/home/yuandy/COCO_dataset/guided_saliency2'
pathToFixationMaps = '/home/yuandy/salicon_data/fixation'

# Path to processed data
pathOutputImages = '/home/yuandy/COCO_dataset/processed_data/images320x240'
pathOutputMaps = '/home/yuandy/COCO_dataset/processed_data/saliency320x240'
pathToPickle = '/home/yuandy/COCO_dataset/processed_data/320x240'

# Path to pickles which contains processed data
TRAIN_DATA_DIR = '/home/yuandy/COCO_dataset/processed_data/320x240/fix_trainData.pickle'
VAL_DATA_DIR = '/home/yuandy/COCO_dataset/320x240/fix_validationData.pickle'
TEST_DATA_DIR = '/home/yuandy/COCO_dataset/256x192/testData.pickle'

# Path to vgg16 pre-trained weights
PATH_TO_VGG16_WEIGHTS = '/home/yuandy/guided_saliency/scripts/vgg16.pkl'

# Input image and saliency map size
INPUT_SIZE = (256, 192)

# Directory to keep snapshots
DIR_TO_SAVE = 'home/yuandy/guided_saliency/test_finetune_weights'
