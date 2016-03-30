import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from src.utils import tools
import feature_constants

logger = tools.get_logger(__name__)


@tools.timeit
def setup_model(data, target, algo=RandomForestRegressor, *args, **kwargs):
    logger.info("Preparing model: {}".format(algo.__name__))
    _model = algo(*args, **kwargs)
    logger.info("Training model")
    _model.fit(data, target)
    return _model


def prepare_data_for_kpi(data, kpi, **options):
    logger.info("Preparing data for KIP: {}".format(kpi))
    training = data[[column for column in (set(data.columns) - {kpi})]]
    target = data[kpi]
    return train_test_split(training, target, **options)


def get_best_metrics(importances, metric_names, n=10):
    return sorted(zip(importances, metric_names), reverse=True)[:n]


@tools.timeit
def importances_RandomForestRegressor(kpi):
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

    best_metrics = get_best_metrics(model.feature_importances_, columns, n=len(columns))

    return best_metrics


@tools.timeit
def importances_RandomForestClassifier(kpi):
    logger.info("Loading hammer data")
    # the first three lines do not contain meaningful data: they are dropped
    statistics = pd.read_csv(feature_constants.CSV_FILES["hammer_statistics"]).iloc[3:, :].set_index("timestamp").dropna()
    target_kpi = statistics[[kpi]]
    k = np.array(target_kpi[kpi].values).astype(float)

    def _get_class(series):
        mean = np.mean(series)
        std = np.std(series)

        return [0 if i < mean - 1.5 * std else
                1 if i < mean else
                2 if i < mean + std else
                3 if i < mean + 1.5 * std else
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

    best_metrics = get_best_metrics(model.feature_importances_, columns, n=len(columns))

    return best_metrics


##########################################################################################
if __name__ == '__main__':
    # RFR_importances = importances_RandomForestRegressor("latency")
    # RFC_importances = importances_RandomForestClassifier("latency")
    pass