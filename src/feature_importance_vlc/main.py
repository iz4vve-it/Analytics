"""
Module analysis for vlc kpi
"""
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import LinearSVC

from src.utils import tools
import src.feature_importance_vlc.feature_constants as feature_constants

LOGGER = tools.get_logger(__name__)


@tools.timeit
def setup_model(data, target, algo=RandomForestRegressor, *args, **kwargs):
    """

    :param data: array of training data
    :param target: array of targets
    :param algo: algorithm class
    :param args: arguments for the algorithm
    :param kwargs: keyword arguments for the algorithm
    :return: trained model
    """
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


def get_best_metrics(importance_array, metric_names, max_metrics=10):
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
    score_train = model.score(train, target_train)
    score_test = model.score(test, target_test)
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
def importance_rfr(kpi, max_metrics=10):
    """

    :param kpi: Name of the current kpi
    :param max_metrics: maximum number of metrics to return
    :return: list of the best metrics
    """
    LOGGER.info("Loading hammer data")
    # the first three lines do not contain meaningful data: they are dropped
    statistics = pd.read_csv(
        feature_constants.CSV_FILES["hammer_statistics"]
    ).iloc[3:, :].set_index("timestamp")
    latency = statistics[[kpi]]

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

    data = latency.join([network, ram, scheduler, cpu]).dropna()
    columns = data[[col for col in set(data.columns) - {kpi}]].columns

    train, test, target_train, target_test = prepare_data_for_kpi(data, kpi)
    model = setup_model(train,
                        target_train,
                        n_estimators=feature_constants.N_ESTIMATORS)

    test_model(model, train, target_train, test, target_test)

    best_metrics = get_best_metrics(model.feature_importances_,
                                    columns,
                                    max_metrics=max_metrics)

    return best_metrics


@tools.timeit
def importance_rfc(kpi, n=10):
    """

    :param kpi: Name of the current kpi
    :param n: maximum number of metrics to return
    :return: list of the best metrics
    """
    LOGGER.info("Loading hammer data")
    # the first three lines do not contain meaningful data: they are dropped
    statistics = pd.read_csv(
        feature_constants.CSV_FILES["hammer_statistics"]
    ).iloc[3:, :].set_index("timestamp").dropna()
    target_kpi = statistics[[kpi]]
    k = np.array(target_kpi[kpi].values).astype(float)

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

    target_kpi[kpi] = _get_class(k)

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

    data = target_kpi.join([network, ram, scheduler, cpu]).dropna()
    columns = data[[col for col in set(data.columns) - {kpi}]].columns

    train, test, target_train, target_test = prepare_data_for_kpi(data, kpi)
    model = setup_model(train,
                        target_train,
                        algo=RandomForestClassifier,
                        n_estimators=10)

    test_model(model, train, target_train, test, target_test)

    best_metrics = get_best_metrics(model.feature_importances_,
                                    columns,
                                    max_metrics=n)
    print set(data[kpi])
    return best_metrics


@tools.timeit
def importance_svm(kpi, max_metrics=10):
    """

    :param kpi: Name of the current kpi
    :param max_metrics: maximum number of metrics to return
    :return: list of the best metrics
    """
    LOGGER.info("Loading hammer data")
    # the first three lines do not contain meaningful data: they are dropped
    statistics = pd.read_csv(
        feature_constants.CSV_FILES["hammer_statistics"]
    ).iloc[3:, :].set_index("timestamp").dropna()
    target_kpi = statistics[[kpi]]
    k = np.array(target_kpi[kpi].values).astype(float)

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

    target_kpi[kpi] = _get_class(k)
    print set(target_kpi[kpi].values)

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

    data = target_kpi.join([network, ram, scheduler, cpu]).dropna()
    columns = data[[col for col in set(data.columns) - {kpi}]].columns

    train, test, target_train, target_test = prepare_data_for_kpi(data, kpi)
    model = setup_model(train, target_train, algo=LinearSVC)

    test_model(model, train, target_train, test, target_test)

    coefficients = model.coef_

    for scores in coefficients:
        print "Normalized scores:"
        print get_best_metrics(
            [score / sum(scores) for score in scores],
            columns,
            max_metrics=max_metrics
        )


@tools.timeit
def main():
    """
    Runs the analysis
    :return:
    """
    rfr_importance = importance_rfr("latency", max_metrics=10)
    rfc_importance = importance_rfc("latency", n=10)
    svm_importance = importance_svm("latency")

    print rfr_importance
    print rfc_importance
    print svm_importance

###############################################################################
    main()
