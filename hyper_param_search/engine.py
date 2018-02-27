"""
Run this script to do hyper-parameter search
this script will use the settings defined in search_settings.py
"""

#TODO: find a way to also record training log
#TODO: incorporate reinforcement learning to assist in hyper-param search

import importlib
import sys
from copy import deepcopy
import hyper_param_search.search_settings as setting
from hyper_param_search.utils import create_obj_from_dict, write_record_to_csv

# import the training function
sys.path.insert(0,setting.training_module_dir)
m = importlib.import_module(setting.training_module)
assert hasattr(m,"training")

# loop through
if (setting.search_method == "loopall"):
    #TODO: implement this
    raise NotImplementedError()
else: # besteach
    param_names = setting.search_params.keys()
    first_write = True
    for pname in param_names:
        # fix all others and vary value of `pname`
        for value in setting.search_params[pname]:
            param_dict = deepcopy(setting.default_param)
            param_dict[pname] = value
            # convert dict to an object
            param_obj = create_obj_from_dict(param_dict)

            # run training and save record
            print("Training using params: {}".format(param_dict))
            result_dict = m.training(param_obj)
            write_record_to_csv(param_dict=param_dict,result_dict=result_dict,path=setting.save_result_to,first_write=first_write)
            first_write = False

class LoopEngine():
    def __init__(self,default_params,tune_params):
        self.default_params = default_params
        self.tune_params = tune_params

    def get_params(self):
        "generator"