import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

from mothernet.model_builder import load_model
from mothernet.models.mothernet_additive import bin_data


def extract_additive_model(model, X_train, y_train, device="cpu", inference_device="cpu"):
    if "cuda" in inference_device and device == "cpu":
        raise ValueError("Cannot run inference on cuda when model is on cpu")
    n_classes = len(np.unique(y_train))
    n_features = X_train.shape[1]

    ys = torch.Tensor(y_train).to(device)
    xs = torch.Tensor(X_train).to(device)

    if X_train.shape[1] > 100:
        raise ValueError("Cannot run inference on data with more than 100 features")
    x_all_torch = torch.concat([xs, torch.zeros((X_train.shape[0], 100 - X_train.shape[1]), device=device)], axis=1)
    X_onehot, bin_edges = bin_data(x_all_torch, n_bins=model.n_bins)
    # why need :len?
    x_src = model.encoder(X_onehot.unsqueeze(1)[:len(X_train)].float())
    y_src = model.y_encoder(ys.unsqueeze(1).unsqueeze(-1))
    train_x = x_src + y_src
    output = model.transformer_encoder(train_x)
    weights, biases = model.decoder(output, ys)
    w = weights.squeeze()[:n_features, :, :n_classes]
    b = biases.squeeze()[:n_classes]
    bins_data_space = bin_edges[:n_features]
    # remove extra classes on output layer
    if inference_device == "cpu":
        def detach(x):
            return x.detach().cpu().numpy()
    else:
        def detach(x):
            return x.detach()

    return detach(w), detach(b), detach(bins_data_space)


def predict_with_additive_model(X_train, X_test, weights, biases, bin_edges, inference_device="cpu", n_bins=64):
    if inference_device == "cpu":
        # FIXME replacing nan with 0 as in TabPFN
        X_test = np.nan_to_num(X_test, 0)
        out = np.zeros((X_test.shape[0], weights.shape[-1]))
        for col, bins, w in zip(X_test.T, bin_edges, weights):
            binned = np.searchsorted(bins, col)
            out += w[binned]
        out += biases
        if np.isnan(out).any():
            print("NAN")
            import pdb
            pdb.set_trace()
        from scipy.special import softmax
        return softmax(out / .8, axis=1)
    elif "cuda" in inference_device:
        mean = torch.Tensor(np.nanmean(X_train, axis=0)).to(inference_device)
        std = torch.Tensor(np.nanstd(X_train, axis=0, ddof=1) + .000001).to(inference_device)
        # FIXME replacing nan with 0 as in TabPFN
        X_train = np.nan_to_num(X_train, 0)
        X_test = np.nan_to_num(X_test, 0)
        std[torch.isnan(std)] = 1
        X_test_scaled = (torch.Tensor(X_test).to(inference_device) - mean) / std
        out = torch.clamp(X_test_scaled, min=-100, max=100)
        import pdb
        pdb.set_trace()
        raise NotImplementedError
        layers = None
        for i, (b, w) in enumerate(layers):
            out = torch.matmul(out, w) + b
            if i != len(layers) - 1:
                out = torch.relu(out)
        return torch.nn.functional.softmax(out / .8, dim=1).cpu().numpy()
    else:
        raise ValueError(f"Unknown inference_device: {inference_device}")


class MotherNetAdditiveClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, path=None, device="cpu", inference_device="cpu"):
        self.path = path
        self.device = device
        self.inference_device = inference_device

    def fit(self, X, y):
        self.X_train_ = X
        le = LabelEncoder()
        y = le.fit_transform(y)
        model, config = load_model(self.path, device=self.device)
        if "model_type" not in config:
            config['model_type'] = config.get("model_maker", 'tabpfn')
        if config['model_type'] != "additive":
            raise ValueError(f"Incompatible model_type: {config['model_type']}")
        model.to(self.device)
        w, b, bin_edges = extract_additive_model(model, X, y, device=self.device, inference_device=self.inference_device)
        self.w_ = w
        self.b_ = b
        self.bin_edges_ = bin_edges
        self.classes_ = le.classes_
        return self

    def predict_proba(self, X):
        return predict_with_additive_model(self.X_train_, X, self.w_, self.b_, self.bin_edges_, inference_device=self.inference_device)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
