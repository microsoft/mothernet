import numpy as np
import torch
from tabpfn.utils import default_device, normalize_data
from tabpfn.priors.utils import get_batch_to_dataloader


def sample_boolean_data_enumerate(hyperparameters, seq_len, num_features):
    # unused, might be better? unclear.
    rank = np.random.randint(1, min(10, num_features))
    grid = torch.meshgrid([torch.tensor([-1, 1])] * num_features)
    inputs = torch.stack(grid, dim=-1).view(-1, num_features)
    outputs = torch.zeros(2**num_features, dtype=bool)

    while 3 * torch.sum(outputs) < len(inputs):
        selected_bits = torch.multinomial(torch.ones(num_features), rank, replacement=False)
        signs = torch.randint(2, (rank,))*2-1
        outputs = outputs + ((signs * inputs[:, selected_bits]) == 1).all(dim=1)
    return (inputs + 1) / 2, outputs

class BooleanConjunctionSampler:
    # This is a class mostly for debugging purposes
    # the object stores the sampled hyperparameters
    def __init__(self, hyperparameters, seq_len, num_features, device):
        self.num_features = num_features
        self.seq_len = seq_len
        self.device = device
        self.max_rank = hyperparameters.get("max_rank", 10)
        self.verbose = hyperparameters.get("verbose", False)

    def sample(self):
        # num_features is always 100, i.e. the number of inputs of the transformer model
        # num_features_active is the number of synthetic datasets features
        # num_features_important is the number of features that actually determine the output
        num_features_active = np.random.randint(1, self.num_features) if self.num_features > 1 else 1
        num_features_important = np.random.randint(1, num_features_active) if num_features_active > 1 else 1
        rank = np.random.randint(1, min(self.max_rank, num_features_important)) if num_features_important > 1 else 1
        num_terms = 0
        features_in_terms = torch.zeros(num_features_important, dtype=bool, device=self.device)
        inputs = 2 * torch.randint(0, 2, (self.seq_len, num_features_active), device=self.device) - 1
        important_indices = torch.randperm(num_features_active)[:num_features_important]
        inputs_important = inputs[:, important_indices]
        outputs = torch.zeros(self.seq_len, dtype=bool, device=self.device)
        while 3 * torch.sum(outputs) < len(inputs):
            selected_bits = torch.multinomial(torch.ones(num_features_important, device=self.device), rank, replacement=False)
            features_in_terms[selected_bits] = True
            signs = torch.randint(2, (rank,), device=self.device) * 2 - 1
            outputs = outputs + ((signs * inputs_important[:, selected_bits]) == 1).all(dim=1)
            num_terms += 1
        sample_params = {'num_terms': num_terms, 'features_in_terms': features_in_terms, 'rank': rank, 'important_indices': important_indices,
                         'num_features_active': num_features_active, 'num_features_important': num_features_important, 'num_features': self.num_features}
        return inputs, outputs, sample_params
    
    def normalize_and_pad(self, x, y):
        x = torch.cat([x, torch.zeros(x.shape[0], self.num_features - x.shape[1], device=self.device)], dim=1)
        xs, ys =  ((x + 1) / 2).unsqueeze(1), y.int().unsqueeze(1).unsqueeze(2)
        xs = normalize_data(xs)
        return xs, ys
    
    def __call__(self):
        x, y, sample_params, = self.sample()
        return *self.normalize_and_pad(x, y), sample_params


def get_batch(batch_size, seq_len, num_features, hyperparameters, device=default_device, num_outputs=1, sampling='normal', epoch=None, **kwargs):
    assert num_outputs == 1
    sample = [BooleanConjunctionSampler(hyperparameters, seq_len=seq_len, num_features=num_features, device=device)() for _ in range(0, batch_size)]
    x, y, _ = zip(*sample)
    y = torch.cat(y, 1).detach().squeeze(2)
    x = torch.cat(x, 1).detach()

    return x, y, y


DataLoader = get_batch_to_dataloader(get_batch)