import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUT_DIR = os.path.join(BASE_DIR, 'out')

CHECKPOINTS_DIR = os.path.join(DATA_DIR, 'checkpoints')
VALIDATION_DATA_DIR = os.path.join(DATA_DIR, 'tfrecords')
image_test_dir = os.path.join(DATA_DIR, 'tfrecords')

RUNS_DIR = os.path.join(OUT_DIR, 'runs')
ATTACKED_OUT_DIR = os.path.join(OUT_DIR, 'attacked')
PREPROCESSED_OUT_DIR = os.path.join(OUT_DIR, 'preprocessed')

ATTACKED_TFRECORD_FILENAME = 'attacked.tfrecord'
PREPROCESSED_TFRECORD_FILENAME = 'preprocessed.tfrecord'
ACCURACY_NPZ_FILENAME = 'accuracy.npz'
TOP5_ACCURACY_NPZ_FILENAME = 'top5_accuracy.npz'
NORMALIZED_L2_DISTANCE_NPZ_FILENAME = 'normalized_l2_distance.npz'

NUM_SAMPLES_VALIDATIONSET = 50000
RESNET_IMAGE_SIZE = 256

if __name__ == '__main__':
    print('BASE_DIR: ', BASE_DIR)
    print('DATA_DIR: ', DATA_DIR)
    print('OUT_DIR: ', OUT_DIR)
    print('CHECKPOINTS_DIR: ', CHECKPOINTS_DIR)
    print('VALIDATION_DATA_DIR: ', VALIDATION_DATA_DIR)
    print('RUNS_DIR: ', RUNS_DIR)
    print('ATTACKED_OUT_DIR: ', ATTACKED_OUT_DIR)
    print('PREPROCESSED_OUT_DIR: ', PREPROCESSED_OUT_DIR)
    print('ATTACKED_TFRECORD_FILENAME: ', ATTACKED_TFRECORD_FILENAME)
    print('PREPROCESSED_TFRECORD_FILENAME: ', PREPROCESSED_TFRECORD_FILENAME)
    print('ACCURACY_NPZ_FILENAME: ', ACCURACY_NPZ_FILENAME)
    print('NORMALIZED_L2_DISTANCE_NPZ_FILENAME: ', NORMALIZED_L2_DISTANCE_NPZ_FILENAME)
    print('NUM_SAMPLES_VALIDATIONSET: ', NUM_SAMPLES_VALIDATIONSET)
    print('RESNET_IMAGE_SIZE: ', RESNET_IMAGE_SIZE)
    
