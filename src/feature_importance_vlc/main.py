"""
Module analysis for vlc kpi

Algorithms to determine feature importance wrt to KPIs
"""
# disabling: no-member, maybe-no-member and unbalanced tuple unpacking
# that are caused by numpy
# pylint: disable=E1101, E1103, W0632
import collections
import itertools
import numpy as np
import operator
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from src.utils import tools
import src.feature_importance_vlc.feature_constants as feature_constants

LOGGER = tools.get_logger(__name__)


@tools.timeit
def load_data(kpi):
    """
    Function to load data from the metrics csv files and hammer file

    :param kpi: name of the kpi to analyze
    :return: dataframe containing data
    """
    LOGGER.info("Loading hammer data")
    # the first three lines do not contain meaningful data: they are dropped
    statistics = pd.read_csv(
        feature_constants.CSV_FILES["hammer_statistics"]
    ).iloc[3:, :].set_index("timestamp")
    kpi_data = statistics[[kpi]]

    LOGGER.info("Loading metrics")

    network = pd.read_csv(
        feature_constants.CSV_FILES["net_interfaces"]
    ).set_index("timestamp")

    ram = pd.read_csv(
        feature_constants.CSV_FILES["proc_stat_meminfo"]
    ).set_index("timestamp")

    scheduler = pd.read_csv(
        feature_constants.CSV_FILES["proc_schedstat"]
    ).set_index("timestamp")

    cpu = pd.read_csv(
        feature_constants.CSV_FILES["proc_stat_cpu"]
    ).set_index("timestamp")

    data = kpi_data.join([network, ram, scheduler, cpu]).dropna()

    return data


DATA = load_data(feature_constants.CURRENT_KPI)


def _get_class(series):
    """
    Converts a series into a set of classes
    """
    mean = np.mean(series)
    std = np.std(series)

    return [0 if i < mean - std else
            1 if i < mean - 0.5 * std else
            2 if i < mean else
            3 if i < mean + 0.5 * std else
            4 for i in series]


@tools.timeit
def setup_model(data, target, algo=None, *args, **kwargs):
    """

    :param data: array of training data
    :param target: array of targets
    :param algo: algorithm class
    :param args: arguments for the algorithm
    :param kwargs: keyword arguments for the algorithm
    :return: trained model
    """
    algo = RandomForestRegressor if algo is None else algo
    LOGGER.info("Preparing model: {}".format(algo.__name__))
    _model = algo(*args, **kwargs)
    LOGGER.info("Training model")
    _model.fit(data, target)
    return _model


def prepare_data_for_kpi(data, kpi, **options):
    """

    :param data: training data
    :param kpi: name of the current kpi
    :param options: options of the split
    :return: split data for training/testing
    """
    LOGGER.info("Preparing data for KPI: {}".format(kpi))
    training = data[[column for column in set(data.columns) - {kpi}]]
    target = data[kpi]
    return train_test_split(training, target, **options)


def get_best_features(importance_array, metric_names, max_metrics=10):
    """
    :param importance_array:
    :param metric_names:
    :param max_metrics: maximum number of metrics to return
    :return: list of most important metrics
    """
    if max_metrics == "all":
        max_metrics = len(importance_array)
    return sorted(
        zip(importance_array, metric_names), reverse=True
    )[:max_metrics]


def test_model(model, train, target_train, test, target_test):
    """

    :param model: trained model
    :param train: training dataset
    :param target_train: target used for training
    :param test: test dataset
    :param target_test: target for test
    """
    try:
        score_train = model.score(train, target_train)
        score_test = model.score(test, target_test)
    except AttributeError:
        LOGGER.info("Model {} has no scoring attribute".format(model))
    else:
        LOGGER.info(
            "Model scoring: training - {:.3%}, test - {:.3%}".format(
                score_train, score_test
            )
        )
        if abs(score_test - score_test) > feature_constants.SCORE_THRESHOLD:
            LOGGER.critical(
                "Model performance very different between training and test"
            )


@tools.timeit
def importance_rfr(data, kpi, max_features=10):
    """
    :param data: dataframe containing training data
    :param kpi: Name of the current kpi
    :param max_features: maximum number of metrics to return
    :return: list of the best metrics
    """

    columns = data[[col for col in set(data.columns) - {kpi}]].columns

    train, test, target_train, target_test = prepare_data_for_kpi(data, kpi)
    model = setup_model(train,
                        target_train,
                        n_estimators=feature_constants.N_ESTIMATORS,
                        max_features=feature_constants.TREES_FEATURES_MODE)

    test_model(model, train, target_train, test, target_test)

    best_metrics = get_best_features(model.feature_importances_,
                                     columns,
                                     max_metrics=max_features)

    return best_metrics


@tools.timeit
def importance_rfc(data, kpi, max_features=10, **kwargs):
    """
    :param data: dataframe containing training data
    :param kpi: Name of the current kpi
    :param max_features: maximum number of metrics to return
    :return: list of the best metrics
    """
    target_kpi = data[[kpi]]
    k = np.array(target_kpi[kpi].values).astype(float)

    target_kpi[kpi] = _get_class(k)

    columns = data[[col for col in set(data.columns) - {kpi}]].columns

    data[kpi] = target_kpi

    train, test, target_train, target_test = prepare_data_for_kpi(data, kpi)
    model = setup_model(train,
                        target_train,
                        algo=RandomForestClassifier,
                        n_estimators=feature_constants.N_ESTIMATORS,
                        max_features=feature_constants.TREES_FEATURES_MODE,
                        **kwargs)

    test_model(model, train, target_train, test, target_test)

    best_metrics = get_best_features(model.feature_importances_,
                                     columns,
                                     max_metrics=max_features)
    return best_metrics


@tools.timeit
def importance_tree_classifier(data, kpi, max_features=10, **kwargs):
    """
    :param data: dataframe containing training data
    :param kpi: Name of the current kpi
    :param max_features: maximum number of metrics to return
    :return: list of the best metrics
    """
    target_kpi = data[[kpi]]
    k = np.array(target_kpi[kpi].values).astype(float)

    target_kpi[kpi] = _get_class(k)

    columns = data[[col for col in set(data.columns) - {kpi}]].columns

    data[kpi] = target_kpi

    train, test, target_train, target_test = prepare_data_for_kpi(data, kpi)
    model = setup_model(train,
                        target_train,
                        algo=DecisionTreeClassifier,
                        **kwargs)

    test_model(model, train, target_train, test, target_test)

    best_metrics = get_best_features(model.feature_importances_,
                                     columns,
                                     max_metrics=max_features)
    return best_metrics


@tools.timeit
def importance_tree_regressor(data, kpi, max_features=10):
    """
    :param data: dataframe containing training data
    :param kpi: Name of the current kpi
    :param max_features: maximum number of metrics to return
    :return: list of the best metrics
    """

    columns = data[[col for col in set(data.columns) - {kpi}]].columns

    train, test, target_train, target_test = prepare_data_for_kpi(data, kpi)
    model = setup_model(train,
                        target_train,
                        algo=DecisionTreeRegressor)

    test_model(model, train, target_train, test, target_test)

    best_metrics = get_best_features(model.feature_importances_,
                                     columns,
                                     max_metrics=max_features)

    return best_metrics


@tools.timeit
def importance_svm(data, kpi, max_features=10, scale=True):
    """
    :param data: dataframe containing training data
    :param kpi: Name of the current kpi
    :param max_features: maximum number of metrics to return
    :param scale: scales scores in [0, 1] if True
    :return: list of the best metrics
    """
    target_kpi = data[[kpi]]
    k = np.array(target_kpi[kpi].values).astype(float)

    target_kpi[kpi] = _get_class(k)

    columns = data[[col for col in set(data.columns) - {kpi}]].columns

    if max_features == "all":
        max_features = len(columns)

    data[kpi] = target_kpi

    train, test, target_train, target_test = prepare_data_for_kpi(data, kpi)
    model = setup_model(train, target_train, algo=LinearSVC)

    test_model(model, train, target_train, test, target_test)

    coefficients = model.coef_
    scores = {}
    for _class, score in enumerate(coefficients):
        if scale:
            score_min = np.min(score)
            score_max = np.max(score)

            norm = (score - score_min) / (score_max - score_min)
            scores[_class] = sorted(zip(norm / sum(norm),
                                        columns),
                                    reverse=True)[: max_features]
        else:
            scores[_class] = sorted(zip(score, columns),
                                    reverse=True)[: max_features]

    return scores


###############################################################################
@tools.timeit
def main():
    """
    Performs the analysis
    """
    rfr_importance = importance_rfr(DATA, feature_constants.CURRENT_KPI,
                                    max_features=10)

    rfc_gini = importance_rfc(DATA, feature_constants.CURRENT_KPI,
                              max_features=10)

    rfc_entropy = importance_rfc(DATA, feature_constants.CURRENT_KPI,
                                 max_features=10,
                                 criterion="entropy")

    dec_tree_gini = importance_tree_classifier(
        DATA,
        feature_constants.CURRENT_KPI,
        max_features=10
    )

    dec_tree_entropy = importance_tree_classifier(
        DATA,
        feature_constants.CURRENT_KPI,
        max_features=10,
        criterion="entropy"
    )

    dec_tree_regressor_importance = importance_tree_regressor(
        DATA,
        feature_constants.CURRENT_KPI,
        max_features=10
    )

    svm_importance = importance_svm(DATA, feature_constants.CURRENT_KPI)
    svm_importance = {
        name for _, name in itertools.chain(*svm_importance.values())
    }
    svm_importance = [("placeholder", name) for name in svm_importance]

    # all features together
    relevant_features = list(itertools.chain(
        rfr_importance,
        rfc_gini,
        rfc_entropy,
        dec_tree_gini,
        dec_tree_regressor_importance,
        dec_tree_entropy,
        svm_importance
    ))

    feature_counter = collections.Counter(name for _, name in relevant_features)
    relevant_features = sorted(feature_counter.items(),
                               key=operator.itemgetter(1),
                               reverse=True)

    with open("features.txt", "w") as path:
        msg = "Features ranked by number of votes (best 10) wrt to KPI: {}\n"
        path.write(msg.format(feature_constants.CURRENT_KPI))
        for i, (name, count) in enumerate(relevant_features, start=1):
            if i > 10:
                break
            if count > 2:
                path.write("{:>3}. {:<35}{:^20}{:>4}\n".format(i,
                                                               name,
                                                               "=>",
                                                               count))


###############################################################################
if __name__ == '__main__':
    main()
