import torch
import torch.nn as nn
import torch.nn.functional as F


def sklearn_like_binning(bin_edges: torch.Tensor, n_bins: int, batch_size: int, num_features: int, data_nona):
    # Bin like in sklearn if there are not enough unique values
    # https://github.com/scikit-learn/scikit-learn/blob/3ee60a720aab3598668af3a3d7eb01d6958859be/sklearn/ensemble/_hist_gradient_boosting/binning.py#L53
    for batch_idx in range(batch_size):
        for col_idx in range(num_features):
            unique_vals = data_nona[col_idx, batch_idx, :].unique(sorted=True).flatten()
            if len(unique_vals) == 1:
                # If all values are the same, we can't bin
                bin_edges[col_idx, batch_idx] = unique_vals
            elif len(unique_vals) < n_bins:
                bin_edges_cat = (unique_vals[:-1] + unique_vals[1:]) * 0.5
                # Fill up the bin edges with inf up until n_bins
                bin_edges_cat = torch.cat((bin_edges_cat, torch.full((n_bins - len(bin_edges_cat) - 1,), torch.inf, device=bin_edges.device)))
                bin_edges[col_idx, batch_idx] = bin_edges_cat
            else:
                pass


def bin_data(data, n_bins, nan_bin=False, single_eval_pos=None, sklearn_binning: bool = False):
    if nan_bin:
        # data is samples x batch x features
        quantiles = torch.arange(n_bins, device=data.device) / (n_bins - 1)

        # Compute quantiles without nan data
        if single_eval_pos is None:
            bin_edges = torch.nanquantile(data, quantiles[1:-1], dim=0)
        else:
            bin_edges = torch.nanquantile(data[:single_eval_pos], quantiles[1:-1], dim=0)

        bin_edges = bin_edges.transpose(0, -1).contiguous()
        data = data.transpose(0, -1).contiguous()

        # Keep track of the nan positions in the data
        isnan = torch.isnan(data)

        # Fill NaNs in order to bin the data.
        data = torch.nan_to_num(data, nan=0.0)
        batch_size, num_features = data.shape[1], data.shape[0]
        if sklearn_binning:
            sklearn_like_binning(bin_edges, n_bins - 1, batch_size, num_features, data)
        X_binned = torch.searchsorted(bin_edges, data)

        # Put NaN data on the last bin.
        X_binned[isnan] = n_bins - 1
        X_onehot = nn.functional.one_hot(X_binned.transpose(0, -1), num_classes=n_bins)
    else:
        # data is samples x batch x features
        data_nona = torch.nan_to_num(data, nan=0)
        quantiles = torch.arange(n_bins + 1, device=data.device) / n_bins
        if single_eval_pos is None:
            bin_edges = torch.quantile(data_nona, quantiles[1:-1], dim=0)
        else:
            bin_edges = torch.quantile(data_nona[:single_eval_pos], quantiles[1:-1], dim=0)
        zero_padding = (data_nona == 0).all(axis=0)
        # FIXME extra data copy
        bin_edges = bin_edges.transpose(0, -1).contiguous()
        data_nona = data_nona.transpose(0, -1).contiguous()

        batch_size, num_features = data_nona.shape[1], data_nona.shape[0]

        if sklearn_binning:
            sklearn_like_binning(bin_edges, n_bins, batch_size, num_features, data_nona)

        X_binned = torch.searchsorted(bin_edges, data_nona)
        X_onehot = nn.functional.one_hot(X_binned.transpose(0, -1), num_classes=n_bins)
        # mask zero padding data
        X_onehot[:, zero_padding, :] = 0
    return X_onehot, bin_edges
