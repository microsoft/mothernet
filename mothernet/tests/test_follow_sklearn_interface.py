import pickle

import numpy as np

from mothernet.prediction import TabPFNClassifier, MotherNetClassifier, MotherNetAdditiveClassifier
from mothernet.evaluation.baselines.distill_mlp import DistilledTabPFNMLP
from mothernet.utils import get_mn_model

from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def test_follow_sklearn_interface():
    xs = np.random.rand(100, 99)
    ys = np.random.randint(0, 3, 100)

    eval_position = xs.shape[0] // 2
    train_xs, train_ys = xs[0:eval_position], ys[0:eval_position]
    test_xs, _ = xs[eval_position:], ys[eval_position:]

    classifier = TabPFNClassifier(device='cpu')
    classifier.fit(train_xs, train_ys)
    print(classifier)  # this might fail in some scenarios
    pred1 = classifier.predict_proba(test_xs)
    pickle_dump = pickle.dumps(classifier)
    classifier = pickle.loads(pickle_dump)
    pred2 = classifier.predict_proba(test_xs)
    assert (pred1 == pred2).all()


def test_our_tabpfn():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model_string = "tabpfn_nooptimizer_emsize_512_nlayers_12_steps_2048_bs_32ada_lr_0.0001_1_gpu_07_24_2023_01_43_33"
    epoch = "1650"
    get_mn_model(f"{model_string}_epoch_{epoch}.cpkt")
    classifier = TabPFNClassifier(device='cpu', model_string=model_string, epoch=epoch)
    classifier.fit(X_train, y_train)
    print(classifier)
    prob = classifier.predict_proba(X_test)
    assert (prob.argmax(axis=1) == classifier.predict(X_test)).all()
    assert classifier.score(X_test, y_test) > 0.9


def test_mothernet_paper():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model_string = "mn_d2048_H4096_L2_W32_P512_1_gpu_warm_08_25_2023_21_46_25_epoch_3940_no_optimizer.pickle"
    model_path = get_mn_model(model_string)
    classifier = MotherNetClassifier(device='cpu', path=model_path)
    classifier.fit(X_train, y_train)
    print(classifier)
    prob = classifier.predict_proba(X_test)
    assert (prob.argmax(axis=1) == classifier.predict(X_test)).all()
    assert classifier.score(X_test, y_test) > 0.9


def test_mothernet_no_model_passed():
    # this will also use the "paper" model above.
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    classifier = MotherNetClassifier(device='cpu')
    classifier.fit(X_train, y_train)
    print(classifier)
    prob = classifier.predict_proba(X_test)
    assert (prob.argmax(axis=1) == classifier.predict(X_test)).all()
    assert classifier.score(X_test, y_test) > 0.9


def test_mothernet_jan_2024():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model_string = "mn_P512_SFalse_L2_1_gpu_01_24_2024_00_31_59_epoch_3950.cpkt"
    model_path = get_mn_model(model_string)
    classifier = MotherNetClassifier(device='cpu', path=model_path)
    classifier.fit(X_train, y_train)
    print(classifier)
    prob = classifier.predict_proba(X_test)
    assert (prob.argmax(axis=1) == classifier.predict(X_test)).all()
    assert classifier.score(X_test, y_test) > 0.9


def test_additive_mothernet_dense():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model_string = "additive_1_gpu_02_14_2024_16_34_15_epoch_950_fixed2.cpkt"
    model_path = get_mn_model(model_string)
    classifier = MotherNetAdditiveClassifier(device='cpu', path=model_path)
    classifier.fit(X_train, y_train)
    print(classifier)
    prob = classifier.predict_proba(X_test)
    assert (prob.argmax(axis=1) == classifier.predict(X_test)).all()
    assert classifier.score(X_test, y_test) > 0.9


def test_baam():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model_string = "baam_H512_Dclass_average_e128_nsamples500_numfeatures20_padzerosFalse_03_14_2024_15_03_22_epoch_400.cpkt"
    model_path = get_mn_model(model_string)
    classifier = MotherNetAdditiveClassifier(device='cpu', path=model_path)
    classifier.fit(X_train, y_train)
    print(classifier)
    prob = classifier.predict_proba(X_test)
    assert (prob.argmax(axis=1) == classifier.predict(X_test)).all()
    assert classifier.score(X_test, y_test) > 0.9


def test_distilled_mlp_paper():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model_string = "tabpfn_nooptimizer_emsize_512_nlayers_12_steps_2048_bs_32ada_lr_0.0001_1_gpu_07_24_2023_01_43_33"
    epoch = "1650"
    get_mn_model(f"{model_string}_epoch_{epoch}.cpkt")
    classifier = make_pipeline(StandardScaler(),
                               DistilledTabPFNMLP(n_epochs=100, device='cpu', hidden_size=128, n_layers=2, dropout_rate=.1, learning_rate=0.01,
                                                  model_string=model_string,
                                                  epoch=epoch, N_ensemble_configurations=3))
    classifier.fit(X_train, y_train)
    print(classifier)
    prob = classifier.predict_proba(X_test)
    assert (prob.argmax(axis=1) == classifier.predict(X_test)).all()
    assert classifier.score(X_test, y_test) > 0.9