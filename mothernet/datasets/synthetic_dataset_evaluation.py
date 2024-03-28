from sklearn.model_selection import train_test_split

from mothernet.datasets import linear_correlated_logistic_regression, linear_correlated_step_function
from mothernet.evaluation.plot_shape_function import plot_shape_function
from mothernet.prediction import MotherNetAdditiveClassifier
from mothernet.utils import get_mn_model


def logistic_regression(model_string: str):
    X, y = linear_correlated_logistic_regression(
        n_features=3, n_tasks=1, n_datapoints=1000, sampling_correlation=0.0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model_path = get_mn_model(model_string)
    classifier = MotherNetAdditiveClassifier(device='cpu', path=model_path)
    classifier.fit(X_train, y_train)

    # Plot shape function
    bin_edges = classifier.bin_edges_
    w = classifier.w_
    plot_shape_function(bin_edges, w)
    print(classifier)
    prob = classifier.predict_proba(X_test)
    assert (prob.argmax(axis=1) == classifier.predict(X_test)).all()
    assert classifier.score(X_test, y_test) > 0.9


def step_function(model_string: str):
    X, y = linear_correlated_step_function(
        n_features=2, n_tasks=1, n_datapoints=10000, sampling_correlation=0.0, plot=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model_path = get_mn_model(model_string)
    classifier = MotherNetAdditiveClassifier(device='cpu', path=model_path)
    classifier.fit(X_train, y_train)

    # Plot shape function
    bin_edges = classifier.bin_edges_
    w = classifier.w_
    plot_shape_function(bin_edges, w)
    prob = classifier.predict_proba(X_test)
    assert (prob.argmax(axis=1) == classifier.predict(X_test)).all()
    assert classifier.score(X_test, y_test) > 0.9


if '__main__' == __name__:
    model_string = "baam_H512_Dclass_average_e128_nsamples500_numfeatures20_padzerosFalse_03_14_2024_15_03_22_epoch_400.cpkt"
    logistic_regression(model_string)
    step_function(model_string)
