"""
Module analysis for vlc kpi

Algorithms to determine feature importance wrt to KPIs
"""
# disabling: no-member, maybe-no-member and unbalanced tuple unpacking
# that are caused by numpy
# pylint: disable=E1101, E1103, W0632
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import f_regression, f_classif, SelectPercentile
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
                        min_samples_leaf=5)

    test_model(model, train, target_train, test, target_test)

    best_metrics = get_best_features(model.feature_importances_,
                                     columns,
                                     max_metrics=max_features)

    return best_metrics


@tools.timeit
def importance_rfc(data, kpi, max_features=10):
    """
    :param data: dataframe containing training data
    :param kpi: Name of the current kpi
    :param max_features: maximum number of metrics to return
    :return: list of the best metrics
    """
    target_kpi = data[[kpi]]
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

    columns = data[[col for col in set(data.columns) - {kpi}]].columns

    data[kpi] = target_kpi

    train, test, target_train, target_test = prepare_data_for_kpi(data, kpi)
    model = setup_model(train,
                        target_train,
                        algo=RandomForestClassifier,
                        n_estimators=feature_constants.N_ESTIMATORS)

    test_model(model, train, target_train, test, target_test)

    best_metrics = get_best_features(model.feature_importances_,
                                     columns,
                                     max_metrics=max_features)
    return best_metrics


@tools.timeit
def importance_decision_tree_classifier(data, kpi, max_features=10):
    """
    :param data: dataframe containing training data
    :param kpi: Name of the current kpi
    :param max_features: maximum number of metrics to return
    :return: list of the best metrics
    """
    target_kpi = data[[kpi]]
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

    columns = data[[col for col in set(data.columns) - {kpi}]].columns

    data[kpi] = target_kpi

    train, test, target_train, target_test = prepare_data_for_kpi(data, kpi)
    model = setup_model(train,
                        target_train,
                        algo=DecisionTreeClassifier,
                        n_estimators=feature_constants.N_ESTIMATORS)

    test_model(model, train, target_train, test, target_test)

    best_metrics = get_best_features(model.feature_importances_,
                                     columns,
                                     max_metrics=max_features)
    return best_metrics


@tools.timeit
def importance_decision_tree_regressor(data, kpi, max_features=10):
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


def importance_univariate(data, kpi, max_features=10, classification=False):
    """
    :param data: dataframe containing training data
    :param kpi: Name of the current kpi
    :param max_features: maximum number of metrics to return
    :return: list of the best metrics
    """
    target_kpi = data[[kpi]]
    k = np.array(target_kpi[kpi].values).astype(float)

    if classification:
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
        data[kpi] = target_kpi

    columns = data[[col for col in set(data.columns) - {kpi}]].columns

    train, test, target_train, target_test = prepare_data_for_kpi(data, kpi)

    model = SelectPercentile(f_classif if classification else f_regression,
                             percentile=10)

    model.fit(train, target_train)

    test_model(model, train, target_train, test, target_test)

    scores = -np.log10(model.pvalues_)
    scores /= scores.max()

    best_metrics = get_best_features(scores,
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


@tools.timeit
def importance_fregression(data, kpi):
    """
    :param data: dataframe containing training data
    :param kpi: Name of the current kpi
    :return: list of the best metrics
    """
    target_kpi = data[[kpi]]
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

    columns = data[[col for col in set(data.columns) - {kpi}]].columns

    data[kpi] = target_kpi

    train, _, target_train, _ = prepare_data_for_kpi(data, kpi)
    _, p_val = f_regression(train, target_train)

    best_metrics = [(i, j) for i, j in sorted(zip(p_val, columns))
                    if np.isfinite(i)]
    best_metrics.sort(reverse=True)
    return best_metrics


###############################################################################
@tools.timeit
def generate_report():
    """
    Runs the analysis and writes the report
    """

    with open("results.txt", "w") as f:
        rfr_importance = importance_rfr(DATA, feature_constants.CURRENT_KPI,
                                        max_features=10)
        f.write("Algorithm: Random Forest Regressor\n")
        for importance, column in rfr_importance:
            f.write("{:>7.3%}\t{}\t{}\n".format(importance, "=>", column))
        f.write("_" * 80)

        rfc_importance = importance_rfc(DATA, feature_constants.CURRENT_KPI,
                                        max_features=10)
        f.write("\n\nAlgorithm: Random Forest Classifier\n")
        for importance, column in rfc_importance:
            f.write("{:>7.3%}\t{}\t{}\n".format(importance, "=>", column))
        f.write("_" * 80)

        f_regression_importance = importance_fregression(
            DATA,
            feature_constants.CURRENT_KPI
        )
        f.write("\n\nAlgorithm: F Regressor\n")
        for importance, column in f_regression_importance:
            f.write("{:>10.6f}\t{}\t{}\n".format(importance, "=>", column))
        f.write("_" * 80)

        svm_importance = importance_svm(DATA, feature_constants.CURRENT_KPI)
        f.write("\n\nAlgorithm: SVM\n")
        for _class, scores in svm_importance.iteritems():
            f.write("Class {}\n".format(_class))
            for score, column in scores:
                f.write("{:>7.3%}\t{}\t{}\n".format(score, "=>", column))
            f.write("#" * 80)
            f.write("\n\n")


def main():
    rfr_importance = importance_rfr(DATA, feature_constants.CURRENT_KPI,
                                    max_features=10)

    rfc_importance = importance_rfc(DATA, feature_constants.CURRENT_KPI,
                                    max_features=10)

    dec_tree_classifier_importance = importance_rfc(
        DATA,
        feature_constants.CURRENT_KPI,
        max_features=10
    )

    dec_tree_regressor_importance = importance_rfc(
        DATA,
        feature_constants.CURRENT_KPI,
        max_features=10
    )

    univariate_classification_importance = importance_univariate(
        DATA, feature_constants.CURRENT_KPI,
        max_features=10,
        classification=True
    )

    univariate_regression_importance = importance_univariate(
        DATA, feature_constants.CURRENT_KPI,
        max_features=10
    )

    print rfr_importance
    print rfc_importance
    print dec_tree_classifier_importance
    print dec_tree_regressor_importance
    print univariate_classification_importance
    print univariate_regression_importance

###############################################################################
if __name__ == '__main__':
    main()
