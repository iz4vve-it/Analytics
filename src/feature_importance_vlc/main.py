import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor

from src.utils import tools
import feature_constants

logger = tools.get_logger(__name__)


def setup_model(data, target, algo=RandomForestRegressor, *args, **kwargs):
    _model = algo(*args, **kwargs)
    _model.fit(data, target)
    return _model


def prepare_data_for_kpi(data, kpi, **options):
    training = data[[col for col in set(data.columns) - {kpi}]]
    target = data[kpi]
    return train_test_split(training, target, **options)


def main():
    # the first three lines do not contain meaningful data: they are dropped
    statistics = pd.read_csv(feature_constants.CSV_FILES["hammer_statistics"]).iloc[3:, :]
    latency = statistics[["timestamp", "latency"]]

    network = pd.read_csv(feature_constants.CSV_FILES["net_interfaces"])
    ram = pd.read_csv(feature_constants.CSV_FILES["proc_stat_meminfo"])
    scheduler = pd.read_csv(feature_constants.CSV_FILES["proc_schedstat"])
    cpu = pd.read_csv(feature_constants.CSV_FILES["proc_stat_cpu"])

    # print network.head()
    data = latency.set_index("timestamp").join(network.set_index("timestamp")).dropna()

    train, test, target_train, target_test = prepare_data_for_kpi(data, "latency")

    print len(train)
    print len(target_train)
    print len(test)
    print len(target_test)

    # model = setup_model()



##########################################################################################
if __name__ == '__main__':
    main()
