import itertools
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# from sklearn.lda import LDA
from sklearn.svm import LinearSVC
import Orange.data
import Orange.associate as associate

from src.utils import tools
import feature_constants

logger = tools.get_logger(__name__)


def get_support(df):
    """
    A priori
    """
    pp = []
    for cnum in range(1, len(df.columns)+1):
        for cols in itertools.combinations(df, cnum):
            s = df[list(cols)].all(axis=1).sum()
            pp.append([",".join(cols), s])
    sdf = pd.DataFrame(pp, columns=["Pattern", "Support"])
    return sdf


@tools.timeit
def setup_model(data, target, algo=RandomForestRegressor, *args, **kwargs):
    logger.info("Preparing model: {}".format(algo.__name__))
    _model = algo(*args, **kwargs)
    logger.info("Training model")
    _model.fit(data, target)
    return _model


def prepare_data_for_kpi(data, kpi, **options):
    logger.info("Preparing data for KPI: {}".format(kpi))
    training = data[[column for column in (set(data.columns) - {kpi})]]
    target = data[kpi]
    return train_test_split(training, target, **options)


def get_best_metrics(importances, metric_names, n=10):
    return sorted(zip(importances, metric_names), reverse=True)[:n]


def test_model(model, train, target_train, test, target_test):
    score_train = model.score(train, target_train)
    score_test = model.score(test, target_test)
    logger.info("Model scoring: training - {:.3%}, test - {:.3%}".format(score_train, score_test))
    if abs(score_test - score_test) > feature_constants.SCORE_THRESHOLD:
        logger.critical("Model performance significantly different between training and test")


@tools.timeit
def importances_RandomForestRegressor(kpi, n=10):
    logger.info("Loading hammer data")
    # the first three lines do not contain meaningful data: they are dropped
    statistics = pd.read_csv(feature_constants.CSV_FILES["hammer_statistics"]).iloc[3:, :].set_index("timestamp")
    latency = statistics[[kpi]]

    logger.info("Loading metrics")
    network = pd.read_csv(feature_constants.CSV_FILES["net_interfaces"]).set_index("timestamp")
    ram = pd.read_csv(feature_constants.CSV_FILES["proc_stat_meminfo"]).set_index("timestamp")
    scheduler = pd.read_csv(feature_constants.CSV_FILES["proc_schedstat"]).set_index("timestamp")
    cpu = pd.read_csv(feature_constants.CSV_FILES["proc_stat_cpu"]).set_index("timestamp")

    data = latency.join([network, ram, scheduler, cpu]).dropna()
    columns = data[[col for col in set(data.columns) - {kpi}]].columns

    train, test, target_train, target_test = prepare_data_for_kpi(data, kpi)
    model = setup_model(train, target_train, n_estimators=feature_constants.N_ESTIMATORS)

    test_model(model, train, target_train, test, target_test)

    best_metrics = get_best_metrics(model.feature_importances_, columns, n=len(columns))

    return best_metrics


@tools.timeit
def importances_RandomForestClassifier(kpi, n=10):
    logger.info("Loading hammer data")
    # the first three lines do not contain meaningful data: they are dropped
    statistics = pd.read_csv(feature_constants.CSV_FILES["hammer_statistics"]).iloc[3:, :].set_index("timestamp").dropna()
    target_kpi = statistics[[kpi]]
    k = np.array(target_kpi[kpi].values).astype(float)

    def _get_class(series):
        mean = np.mean(series)
        std = np.std(series)

        return [0 if i < mean - std else
                1 if i < mean - 0.5 * std else
                2 if i < mean else
                3 if i < mean + 0.5 * std else
                4 for i in series]

    target_kpi[kpi] = _get_class(k)

    logger.info("Loading metrics")
    network = pd.read_csv(feature_constants.CSV_FILES["net_interfaces"]).set_index("timestamp")
    ram = pd.read_csv(feature_constants.CSV_FILES["proc_stat_meminfo"]).set_index("timestamp")
    scheduler = pd.read_csv(feature_constants.CSV_FILES["proc_schedstat"]).set_index("timestamp")
    cpu = pd.read_csv(feature_constants.CSV_FILES["proc_stat_cpu"]).set_index("timestamp")

    data = target_kpi.join([network, ram, scheduler, cpu]).dropna()
    columns = data[[col for col in set(data.columns) - {kpi}]].columns

    train, test, target_train, target_test = prepare_data_for_kpi(data, kpi)
    model = setup_model(train, target_train, algo=RandomForestClassifier, n_estimators=10)

    test_model(model, train, target_train, test, target_test)

    best_metrics = get_best_metrics(model.feature_importances_, columns, n=len(columns))
    print set(data[kpi])
    return best_metrics


@tools.timeit
def importances_SVM(kpi, n=10):
    logger.info("Loading hammer data")
    # the first three lines do not contain meaningful data: they are dropped
    statistics = pd.read_csv(feature_constants.CSV_FILES["hammer_statistics"]).iloc[3:, :].set_index("timestamp").dropna()
    target_kpi = statistics[[kpi]]
    k = np.array(target_kpi[kpi].values).astype(float)

    def _get_class(series):
        mean = np.mean(series)
        std = np.std(series)

        return [0 if i < mean - std else
                1 if i < mean - 0.5 * std else
                2 if i < mean else
                3 if i < mean + 0.5 * std else
                4 for i in series]

    target_kpi[kpi] = _get_class(k)
    print set(target_kpi[kpi].values)

    logger.info("Loading metrics")
    network = pd.read_csv(feature_constants.CSV_FILES["net_interfaces"]).set_index("timestamp")
    ram = pd.read_csv(feature_constants.CSV_FILES["proc_stat_meminfo"]).set_index("timestamp")
    scheduler = pd.read_csv(feature_constants.CSV_FILES["proc_schedstat"]).set_index("timestamp")
    cpu = pd.read_csv(feature_constants.CSV_FILES["proc_stat_cpu"]).set_index("timestamp")

    data = target_kpi.join([network, ram, scheduler, cpu]).dropna()
    columns = data[[col for col in set(data.columns) - {kpi}]].columns

    train, test, target_train, target_test = prepare_data_for_kpi(data, kpi)
    model = setup_model(train, target_train, algo=LinearSVC)

    test_model(model, train, target_train, test, target_test)

    coefficients = model.coef_

    for scores in coefficients:
        print "Normalized scores:"
        print get_best_metrics([score / sum(scores) for score in scores], columns, n=5)


@tools.timeit
def main():
    RFR_importances = importances_RandomForestRegressor("latency", n=10)
    RFC_importances = importances_RandomForestClassifier("latency", n=10)
    SVM_importances = importances_SVM("latency")

##########################################################################################
if __name__ == '__main__':
    main()
