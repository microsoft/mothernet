import matplotlib.pyplot as plt
import numpy as np


def plot_shape_function(bin_edges: np.ndarray, w: np.ndarray):
    num_classes = w.shape[2]
    num_features = len(bin_edges)
    fig, axs = plt.subplots(num_classes, num_features, figsize=(2*num_features, 2*num_classes),
                            sharex=True, sharey=True)
    for class_idx in range(num_classes):
        for feature_idx in range(num_features):
            axs[class_idx][feature_idx].plot(
                bin_edges[feature_idx], w[feature_idx][0:-1][:, class_idx] - w[feature_idx][0:-1][:, class_idx].mean())
            if class_idx == 0:
                axs[class_idx][feature_idx].set_title(f'Feature {feature_idx}')
            if feature_idx == 0:
                axs[class_idx][feature_idx].set_ylabel(f'Class {class_idx}')
    plt.tight_layout()
    plt.show()