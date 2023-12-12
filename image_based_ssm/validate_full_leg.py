import os
from ssm_utils import *
from ssm_config import *
from ssm_test_surgical_plan import test_ssm_surgical_plan as ssm_eval
import ants


bone = 'left_femur'
cond = 'full_length'

test_dir = TEST_DIRS[bone]
model_dir = RESULTS_DIRS[bone]
warped_test_dir = os.path.join(ROOT_DIR, f'wrapped_label/{bone}/test/{cond}/')
results_dir = os.path.join(RESULTS_DIRS[bone], f'{cond}/')


check_path_exist(warped_test_dir)
check_path_exist(results_dir)

fixed_anatomy = ants.image_read(os.path.join(ROOT_DIR, bone + '.nii.gz'))

# Test model
ssm_eval(model_dir, fixed_anatomy, test_dir, warped_test_dir, results_dir, bone, cond)
            
    