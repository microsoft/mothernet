#!/usr/bin/env python
# coding: utf-8

# ## Setup

# In[2]:


import random
import time
import warnings
from datetime import datetime

import torch

import numpy as np

import matplotlib.pyplot as plt
from scripts.differentiable_pfn_evaluation import eval_model_range
from scripts.model_builder import get_model, get_default_spec, save_model, load_model
from scripts.transformer_prediction_interface import transformer_predict, get_params_from_config, load_model_workflow

from scripts.model_configs import *

from datasets import load_openml_list, open_cc_dids, open_cc_valid_dids
from priors.utils import plot_prior, plot_features
from priors.utils import uniform_int_sampler_f

from scripts.tabular_metrics import calculate_score_per_method, calculate_score
from scripts.tabular_evaluation import evaluate

from priors.differentiable_prior import DifferentiableHyperparameterList, draw_random_style, merge_style_with_info
from scripts import tabular_metrics
from notebook_utils import *


# In[3]:


large_datasets = True
max_samples = 10000 if large_datasets else 5000
bptt = 10000 if large_datasets else 3000
suite='cc'


# In[11]:


device = 'cuda'
base_path = '.'
max_features = 100


# In[12]:


def print_models(model_string):
    print(model_string)

    for i in range(80):
        for e in range(50):
            exists = Path(os.path.join(base_path, f'models_diff/prior_diff_real_checkpoint{model_string}_n_{i}_epoch_{e}.cpkt')).is_file()
            if exists:
                print(os.path.join(base_path, f'models_diff/prior_diff_real_checkpoint{model_string}_n_{i}_epoch_{e}.cpkt'))
        print()


# In[13]:


def train_function(config_sample, i, add_name='', state_dict=None):
    start_time = time.time()
    N_epochs_to_save = 100
    save_every = max(1, config_sample['epochs'] // N_epochs_to_save)
    
    def save_callback(model, epoch):
        if not hasattr(model, 'last_saved_epoch'):
            model.last_saved_epoch = 0
        # if ((time.time() - start_time) / (maximum_runtime * 60 / N_epochs_to_save)) > model.last_saved_epoch:
        print(f"epoch: {epoch* config_sample['epochs']} save_every: {save_every}")
        if (epoch * config_sample['epochs']) % save_every == 0 or model.last_saved_epoch <10:
            file_name = f'models_diff/prior_diff_real_checkpoint{add_name}_n_{i}_epoch_{model.last_saved_epoch}.cpkt'
            print(f'Saving model to {file_name}')
            config_sample['epoch_in_training'] = epoch
            save_model(model, base_path, file_name,
                           config_sample)
            model.last_saved_epoch = model.last_saved_epoch + 1 # TODO: Rename to checkpoint
    
    model = get_model(config_sample
                      , device
                      , should_train=True
                      , verbose=1
                      , epoch_callback = save_callback, state_dict=state_dict)
    
    return model


# ## Define prior settings

# In[14]:


def reload_config(config_type='causal', task_type='multiclass', longer=0):
    config = get_prior_config(config_type=config_type)
    
    config['prior_type'], config['differentiable'], config['flexible'] = 'prior_bag', True, True
    
    model_string = '_featurewise_mlp_hidden_512'
    
    config['epochs'] = 12000
#    config['epochs'] = 1
    config['recompute_attn'] = True

    config['max_num_classes'] = 10
    config['num_classes'] = uniform_int_sampler_f(2, config['max_num_classes'])
    config['balanced'] = False
    model_string = model_string + '_multiclass'
    
    model_string = model_string + '_'+datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    
    return config, model_string


config, model_string = reload_config(longer=1)

path = "models_diff"
filename = "prior_diff_real_checkpointtry_retraining_batch_512_onehot_n_0_epoch_18.cpkt"
# model = get_model(config_sample, device, should_train=True, verbose=1)
config['device'] = 'cuda'
# import pdb; pdb.set_trace()
model_state, optimizer_state, config_sample = torch.load(
    os.path.join(path, filename), map_location='cpu')
module_prefix = 'module.'
model_state = {k.replace(module_prefix, ''): v for k, v in model_state.items()}
# model_old, config_sample =  load_model(path=path, filename=filename, device=config['device'], eval_positions=None, verbose=1)
model_string = "_retraining_batch_2048_lr0001"
#config_sample['emsize'] = 512
config_sample['differentiable_hyperparameters']["prior_mlp_activations"] ={'distribution': 'meta_choice_mixed', 'choice_values': [
            torch.nn.Tanh
            , torch.nn.Identity
            , torch.nn.ReLU
        ]}
config_sample['batch_size'] = 2048
config_sample['epochs'] = 100
config_sample['lr'] = 0.00001
config_sample['num_steps'] = 512 / 32
config_sample['aggregate_k_gradients'] = 32
config_sample['num_classes'] = uniform_int_sampler_f(2, config_sample['max_num_classes'])
config_sample['num_features_used'] = {'uniform_int_sampler_f(3,max_features)': uniform_int_sampler_f(1, max_features)}

config_sample = evaluate_hypers(config_sample)
model = train_function(config_sample, i=0, add_name=model_string, state_dict=model_state)
# import pickle
# with open(f"all_{model_string}.pickle", "wb") as f:
#    pickle.dump(model[:3], f, protocol=-1)
rank = 0
if 'LOCAL_RANK' in os.environ:
    # launched with torch.distributed.launch
    rank = int(os.environ["LOCAL_RANK"])
    print('torch.distributed.launch and my rank is', rank)

if rank == 0:
    i = 0
    save_model(model[2], base_path, f'models_diff/prior_diff_real_checkpoint{model_string}_n_{i}_epoch_on_exit.cpkt',
                   config_sample)

# In[ ]:
