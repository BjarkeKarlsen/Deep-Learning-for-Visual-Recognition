DATASET_NAME = "saberzl/SID_Set"
NUM_CLASSES = 3
CLASS_NAMES = {0: 'Real', 1: 'Synthetic', 2: 'Tampered'}

IMAGE_SIZE = 512
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

EPOCHS = 50
BATCH_SIZE = 20
LEARNING_RATE = 0.001

TRAIN_SAMPLES = 500
VAL_SAMPLES = 100
TEST_SAMPLES = 10
TEST_BATCH_SIZE = 8
NUM_WORKERS = 0

MODEL_PATH = 'models/best_model.pth'
RESULTS_DIR = 'results'

SEED = 42

LOGGING_LEVEL = 'INFO'  # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'

"""Loss configuration dictionary.
You can customize the loss functions and their weights here.
The higher the weight, the more influence that loss has on the total loss.
"""

LOSS_CONFIG = {
        'cls_loss_type': 'crossentropy',
        'seg_loss_type': 'dice',
        'cls_weight': 1.0,
        'seg_weight': 5.0  
    }

    # Another example (Focal + Tversky)
    # LOSS_CONFIG = {
    #     'cls_loss_type': 'focal',
    #     'seg_loss_type': 'tversky',
    #     'cls_weight': 0.7,
    #     'seg_weight': 1.3,
    #     'cls_kwargs': {'alpha': 1, 'gamma': 2},
    #     'seg_kwargs': {'alpha': 0.3, 'beta': 0.7}
    # }