###
### Code from https://github.com/merantix-momentum/concurvity-regularization/blob/main/main/concurvity.py
###

import functorch
import torch


def one_vs_rest(component_vals: torch.Tensor, kind: str) -> torch.Tensor:
    """
    Regulariser based on concurvity definition by
    "The Effect of Concurvity in Generalized Additive Models Linking Mortality to Ambient Particulate Matter"
    by Timothy O. Ramsay, Richard T. Burnett, and Daniel Krewski

    It regularizes the correlation between each additive element and the sum of the remaining elements.

    Shape of component_vals: [num_additive_components, batch_size, *]
    """
    component_vals = component_vals.reshape(len(component_vals), -1)

    sum_xj = component_vals.sum(dim=0)
    sum_less_xi = sum_xj - component_vals
    batched_components = torch.stack([sum_less_xi, component_vals], dim=1)

    if kind == "corr":
        batched_pearson_correlation = functorch.vmap(correlation)
    elif kind == "cov":
        batched_pearson_correlation = functorch.vmap(torch.cov)
    else:
        raise NotImplementedError(f"Unknown kind: {kind}")
    return torch.mean(torch.abs(batched_pearson_correlation(batched_components)[:, 0, 1]))


def pairwise(component_vals: torch.Tensor, kind: str, eps: float) -> torch.Tensor:
    """
    Pairwise Regulariser based on concurvity definition by
    "The Effect of Concurvity in Generalized Additive Models Linking Mortality to Ambient Particulate Matter"
    by Timothy O. Ramsay, Richard T. Burnett, and Daniel Krewski

    It regularizes pairwise orthonormality of the additive components.

    Shape of component_vals: [num_additive_components, batch_size, *]
    """
    component_vals = component_vals.reshape(len(component_vals), -1)
    if kind == "corr":
        matrix = correlation(component_vals, eps=eps)
    elif kind == "cov":
        matrix = torch.cov(component_vals)
    else:
        raise NotImplementedError(f"Unknown kind: {kind}")
    upper_triangular_idx = torch.triu_indices(len(matrix), len(matrix), 1)
    return torch.mean(torch.abs(matrix[upper_triangular_idx[0, :], upper_triangular_idx[1, :]]))


def correlation(z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Computes the correlation matrix of a given tensor.

    This function takes as input a tensor and calculates the correlation matrix. If the standard deviation is zero,
    the correlation is set to zero. A small epsilon is added to the standard deviation to avoid division by zero.

    Args: z (torch.Tensor): A tensor for which the correlation matrix is to be computed. eps (float, optional):
    A small value added to the standard deviation to prevent division by zero. Defaults to 1e-12.

    Returns: torch.Tensor: The correlation matrix of the input tensor. """
    std = torch.std(z, dim=1)
    std = std * std.reshape(-1, 1)
    cc = torch.cov(z) / (std + eps)
    cc = torch.where(std == 0.0, 0.0, cc)
    return cc