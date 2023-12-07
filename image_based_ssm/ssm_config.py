# config.py
import os


BONE_STRUCTURE = ['left_femur', 'right_femur', 'left_tibia', 'right_tibia']
CONDITION = ['proximal_only', 'add_distal', 'reduced_distal']

ROOT_DIR = '/home/rgu/Documents'
OUT_DIR = '/home/rgu/Documents/ssm_results'

TRAIN_DIR = os.path.join(ROOT_DIR, 'UK dataset', 'nnUNet_raw', 'Dataset1000_NMDID', 'labelsTr') 


JSON_PATH = os.path.join(OUT_DIR, 'training.json')

WARPED_DIRS = {}
RESULTS_DIRS = {}
TEST_DIRS = {}
RESULTS_DIRS_TESTING = {}

for bone in BONE_STRUCTURE:
    WARPED_DIRS[bone] = {
        'train': os.path.join(ROOT_DIR, f'wrapped_label/{bone}/train/'), 
    }
    TEST_DIRS[bone] = os.path.join(ROOT_DIR, 'test', bone) # where save all the testing images for surgical planning
    RESULTS_DIRS[bone] = os.path.join(OUT_DIR, bone) # for trained ssm model

    RESULTS_DIRS_TESTING[bone] = {}
    for cond in CONDITION:
        WARPED_DIRS[bone].update({
            cond: os.path.join(ROOT_DIR, f'wrapped_label/{bone}/test/{cond}/'),
        })
        RESULTS_DIRS_TESTING[bone].update({
            cond: os.path.join(RESULTS_DIRS[bone], f'{cond}/'),
        })

