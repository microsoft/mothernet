import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from mothernet.evaluation.concurvity import pairwise
from mothernet.prediction import MotherNetAdditiveClassifier
from mothernet.priors import StepFunctionPrior
from mothernet.utils import get_mn_model


def plot_shape_function(bin_edges: np.ndarray, w: np.ndarray):
    num_classes = w.shape[2]
    num_features = len(bin_edges)
    fig, axs = plt.subplots(num_classes, num_features, figsize=(2 * num_features, 2 * num_classes),
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


def eval_step_function():
    step_function_prior = StepFunctionPrior({'max_steps': 1, 'sampling': 'uniform'})
    X, y, step_function = step_function_prior._get_batch(batch_size=1, n_samples=500, num_features=2)
    X = X.squeeze().numpy()
    y = y.squeeze().numpy()

    # Plot the shape function here
    fig, ax = plt.subplots(ncols=2, sharey=True)
    ax[0].plot(X[:, 0], step_function[0, :, 0], 'o')
    ax[1].plot(X[:, 1], step_function[0, :, 1], 'o')
    ax[0].set_xlabel('Feature 0')
    ax[1].set_xlabel('Feature 1')
    ax[0].set_ylabel('y')
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model_string = "baam_H512_Dclass_average_e128_nsamples500_numfeatures20_padzerosFalse_03_14_2024_15_03_22_epoch_400.cpkt"
    model_path = get_mn_model(model_string)
    classifier = MotherNetAdditiveClassifier(device='cpu', path=model_path)
    classifier.fit(X_train, y_train)
    print(classifier)

    # Plot shape function
    bin_edges = classifier.bin_edges_
    w = classifier.w_
    plot_shape_function(bin_edges, w)

    prob, additive_comp = classifier.predict_proba_with_additive_components(X_test)
    conc = pairwise(torch.from_numpy(np.stack([additive_comp[0][:, 1], additive_comp[1][:, 1]])), kind='corr',
                    eps=1e-12)
    print(f'Concurvity: {conc:.3f}')
    assert (prob.argmax(axis=1) == classifier.predict(X_test)).all()
    assert classifier.score(X_test, y_test) > 0.9


eval_step_function()
