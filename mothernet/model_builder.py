import os
import subprocess as sp

import torch
from torch import nn

import mothernet.models.encoders as encoders
from mothernet.dataloader import get_dataloader
from mothernet.train import train
from mothernet.model_configs import get_model_default_config
from mothernet.models.mothernet_additive import MotherNetAdditive
from mothernet.models.perceiver import TabPerceiver
from mothernet.models.tabpfn import TabPFN
from mothernet.models.biattention_tabpfn import BiAttentionTabPFN
from mothernet.models.biattention_additive_mothernet import BiAttentionMotherNetAdditive
from mothernet.models.mothernet import MotherNet
from mothernet.config_utils import nested_dict

try:
    from functools import cache
except ImportError:
    from functools import lru_cache
    cache = lru_cache(maxsize=None)


def get_criterion(max_num_classes):
    if max_num_classes == 2:
        loss = nn.BCEWthLogitsLoss(reduction='none')
    elif max_num_classes > 2:
        loss = nn.CrossEntropyLoss(reduction='none')
    else:
        raise ValueError(f"Invalid number of classes: {max_num_classes}")
    return loss


def save_model(model, optimizer, scheduler, path, filename, config_sample):
    optimizer_dict = optimizer.state_dict() if optimizer is not None else None

    import cloudpickle
    torch.save((model.state_dict(), optimizer_dict, scheduler, config_sample), os.path.join(path, filename), pickle_module=cloudpickle)


def get_gpu_memory():
    command = "nvidia-smi"
    memory_free_info = sp.check_output(command.split()).decode('ascii')
    return memory_free_info


@cache
def load_model(path, device, verbose=False):
    states = torch.load(path, map_location='cpu')
    model_state = states[0]
    config_sample = states[-1]
    if 'y_encoder' not in config_sample and 'onehot' in str(path):
        # workaround for the single model that was saved without y_encoder
        # that happens to be my reference model.
        config_sample['y_encoder'] = 'one_hot'
    _, model, *_ = get_model(config_sample, device=device, should_train=False, verbose=verbose)
    module_prefix = 'module.'
    model_state = {k.replace(module_prefix, ''): v for k, v in model_state.items()}
    model_state.pop("criterion.weight", None)

    decoder_summary_weights = ["query", "output_layer.q_proj_weight", "output_layer.in_proj_weight", "output_layer.k_proj_weight", "output_layer.v_proj_weight",
                               "output_layer.in_proj_bias", "output_layer.out_proj.weight", "output_layer.out_proj.bias"]
    for weights in decoder_summary_weights:
        full_name = "decoder." + weights
        if full_name in model_state:
            model_state['decoder.summary_layer.' + weights] = model_state.pop(full_name)

    if "encoder.weight" in model_state and "model_type" in config_sample and config_sample['model_type'] == "additive":
        model_state['encoder.1.weight'] = model_state.pop("encoder.weight")
        model_state['encoder.1.bias'] = model_state.pop("encoder.bias")

    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    return model, config_sample


def get_encoder(config):
    if config['model_type'] == "batabpfn":
        return encoders.Linear(1, config['transformer']['emsize'], replace_nan_by_zero=True)
    if ((config['prior']['classification']['nan_prob_no_reason'] > 0.0) or
        (config['prior']['classification']['nan_prob_a_reason'] > 0.0) or
            (config['prior']['classification']['nan_prob_unknown_reason'] > 0.0)):
        encoder = encoders.NanHandlingEncoder(config['prior']['num_features'], config['transformer']['emsize'])
    else:
        encoder = encoders.Linear(config['prior']['num_features'], config['transformer']['emsize'], replace_nan_by_zero=True)
    return encoder


def get_y_encoder(config):
    if config['transformer']['y_encoder'] == 'one_hot':
        y_encoder = encoders.OneHotAndLinear(config['prior']['classification']['max_num_classes'], emsize=config['transformer']['emsize'])
    elif config['transformer']['y_encoder'] == 'linear':
        y_encoder = encoders.Linear(1, emsize=config['transformer']['emsize'])
    else:
        raise ValueError(f"Unknown y_encoder: {config['transformer']['y_encoder']}")
    return y_encoder


def old_config_to_new(old_config, new_config):
    # this is not for restarting learning, only inference, so it doesn't convert orchestration parameters
    # it takes a flat config and converts it to a nested one based on the structure of new_config
    old_config['learning_rate'] = old_config.pop('lr')
    if "bptt" in old_config:
        old_config['n_samples'] = old_config.pop('bptt')
    old_config.update(old_config.pop("differentiable_hyperparameters", {}))
    if "y_encoder" not in old_config:
        old_config['y_encoder'] = 'linear'
    if "decoder_em_size" in old_config:
        old_config['decoder_embed_dim'] = old_config.pop('decoder_em_size')
    if "model_maker" in old_config:
        old_config['model_type'] = old_config.pop('model_maker')
    if "em_size" in old_config:
        old_config['emsize'] = old_config.pop('em_size')
    if "aggregate_gradients" in old_config:
        old_config['aggregate_k_gradients'] = old_config.pop('aggregate_gradients')
    if "model_type" not in old_config:
        old_config['model_type'] = 'tabpfn'
    if "num_predicted_hidden_layers" in old_config:
        old_config['predicted_hidden_layers'] = old_config.pop('num_predicted_hidden_layers')
    if "boolean_p_uninformative" in old_config:
        old_config['p_uninformative'] = old_config.pop('boolean_p_uninformative')
    if "boolean_max_fraction_uninformative" in old_config:
        old_config['max_fraction_uninformative'] = old_config.pop('boolean_max_fraction_uninformative')
    if old_config.pop("special_token", False):
        old_config['decoder_type'] = 'special_token'

    if old_config.pop("prenorm", False):
        print("prenorm is not supported anymore")
    if not old_config.pop("output_attention", True):
        raise NotImplementedError("output_attention=False is not supported anymore")
    if old_config.pop("decoder_two_hidden_layers", False):
        old_config['decoder_hidden_layers'] = 2
    else:
        old_config['decoder_hidden_layers'] = 1

    ignored_configs = ['seq_len_used', 'verbose', 'noise_type', 'normalize_to_ranking', 'normalize_by_used_features', 'num_categorical_features_sampler_a',
                       'differentiable', 'flexible', 'bptt_extra_samples', 'dynamic_batch_size', 'new_mlp_per_example', 'batch_size_per_gp_sample',
                       'normalize_ignore_label_too', 'differentiable_hps_as_style', 'rotate_normalized_labels', 'canonical_y_encoder',
                       'total_available_time_in_s', 'normalize_with_sqrt', 'done_part_in_training', 'mix_activations', 'save_every', 'create_new_run',
                       'perceiver_large_dataset', 'no_double_embedding', 'losses', 'wallclock_times', 'learning_rates', 'experiment', 'base_path',
                       'num_gpus', 'device', 'epoch_in_training', 'hid_factor', 'warm_start_from', 'continue_old_config', 'use_cpu', 'st_checkpoint_dir',
                       'no_mlflow', 'load_file', 'continue_run', 'load_strict', 'restart_scheduler', 'extra_fast_test', 'stop_after_epochs', 'shared_embedding',
                       'n_samples_used', 'double_embedding', 'learing_rate', 'gpu_id', 'agg_gradients', 'boolean_prior', 'seed_everything', 'model-type',
                       'num_features_used', 'max_eval_pos']
    if old_config['model_type'] == 'tabpfn':
        # we used to store mothernet parameters in tabpfn models, but we no longer allow that
        ignored_configs.extend(['decoder_embed_dim', 'decoder_hidden_size', 'predicted_hidden_layer_size',
                                'predicted_hidden_layers', 'weight_embedding_rank', 'decoder_hidden_layers'])
    if old_config['model_type'] in ['mothernet', 'additive']:
        ignored_configs.extend(['num_latents'])
    for k in ignored_configs:
        old_config.pop(k, None)
    translated_config = nested_dict()
    for k, v in new_config.items():
        if k in old_config:
            translated_config[k] = old_config.pop(k)
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, dict):
                    for k3, v3 in v2.items():
                        if k3 in old_config:
                            translated_config[k][k2][k3] = old_config.pop(k3)
                elif k2 in old_config:
                    translated_config[k][k2] = old_config.pop(k2)
    if len(old_config):
        raise ValueError(f"Unknown parameters: {old_config.keys()}")
    return translated_config


def get_model(config, device, should_train=True, verbose=False, model_state=None, optimizer_state=None,
              scheduler=None, epoch_callback=None, load_model_strict=True):
    passed_config = config.copy()

    # backwards compatibility for model names
    if 'model_type' not in passed_config:
        if 'model_maker' in passed_config:
            passed_config['model_type'] = passed_config.pop('model_maker')
        else:
            passed_config['model_type'] = 'tabpfn'

    if passed_config['model_type'] == 'mlp':
        passed_config['model_type'] = 'mothernet'
    config = get_model_default_config(passed_config['model_type'])

    if 'optimizer' not in passed_config:
        passed_config = old_config_to_new(passed_config, config)
    config.update(passed_config)
    verbose_train, verbose_prior = verbose >= 1, verbose >= 2
    config['verbose'] = verbose_prior

    criterion = get_criterion(config['prior']['classification']['max_num_classes'])

    # backwards compatibility for cases where absence of parameter doesn't correspond to current default
    if 'n_samples' not in passed_config['prior']:
        config['prior']['n_samples'] = config['bptt']
    if 'y_encoder' not in passed_config['transformer']:
        config['transformer']['y_encoder'] = 'linear'

    if 'mothernet' in config:
        if 'decoder_activation' not in passed_config['mothernet']:
            config['mothernet']['decoder_activation'] = 'relu'
        if 'decoder_type' not in passed_config['mothernet']:
            config['mothernet']['decoder_type'] = 'output_attention'
        if (passed_config['mothernet'].get('weight_embedding_rank', None) is not None
                and 'low_rank_weights' not in passed_config['mothernet']):
            config['mothernet']['low_rank_weights'] = True

    y_encoder = get_y_encoder(config)

    encoder = get_encoder(config)

    if config['prior']['classification']['max_num_classes'] > 2:
        n_out = config['prior']['classification']['max_num_classes']
    else:
        n_out = 1

    model_type = config['model_type']

    if model_type == "mothernet":
        model = MotherNet(
            encoder, n_out=n_out,
            y_encoder_layer=y_encoder, **config['transformer'], **config['mothernet']
        )
    elif model_type == 'perceiver':
        model = TabPerceiver(
            encoder_layer=encoder, n_out=n_out,
            y_encoder_layer=y_encoder, **config['transformer'], **config['mothernet'], **config['perceiver']
        )
    elif model_type == "additive":
        model = MotherNetAdditive(
            n_out=n_out, n_features=config['prior']['num_features'],
            y_encoder_layer=y_encoder, **config['transformer'], **config['mothernet'], **config['additive'])
    elif model_type == "tabpfn":
        model = TabPFN(
            encoder, n_out=n_out, y_encoder_layer=y_encoder, **config['transformer']
        )
    elif model_type == "batabpfn":
        # FIXME hack
        config['transformer']['nhead'] = 4
        model = BiAttentionTabPFN(
            encoder, n_out=n_out, y_encoder_layer=y_encoder, **config['transformer'], **config['biattention']
        )
    elif model_type == "baam":
        # FIXME hack
        config['transformer']['nhead'] = 4
        model = BiAttentionMotherNetAdditive(
            n_out=n_out, n_features=config['prior']['num_features'],
            y_encoder_layer=y_encoder, **config['transformer'], **config['mothernet'], **config['additive']
        )

    else:
        raise ValueError(f"Unknown model type {model_type}.")

    if model_state is not None:
        if not load_model_strict:
            for k, v in model.state_dict().items():
                if k in model_state and model_state[k].shape != v.shape:
                    model_state.pop(k)
        model.load_state_dict(model_state, strict=load_model_strict)

    if verbose:
        print(f"Using a Transformer with {sum(p.numel() for p in model.parameters())/1000/1000:.{2}f} M parameters")

    if 'losses' in config:
        # for continuing training
        model.losses = config['losses']
        model.learning_rates = config['learning_rates']
        model.wallclock_times = config.get('wallclock_times', [])

    if should_train:
        dl = get_dataloader(prior_config=config['prior'], dataloader_config=config['dataloader'], device=device)
        model = train(dl, model, criterion=criterion, optimizer_state=optimizer_state, scheduler=scheduler,
                      epoch_callback=epoch_callback, verbose=verbose_train, device=device, **config['optimizer'])
    else:
        model = None, model, None, None

    return model
