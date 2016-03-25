import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from src.utils import tools
import feature_constants

logger = tools.get_logger(__name__)


@tools.timeit
def setup_model(data, target, algo=RandomForestRegressor, *args, **kwargs):
    _model = algo(*args, **kwargs)
    _model.fit(data, target)
    return _model


def prepare_data_for_kpi(data, kpi, **options):
    training = data[[col for col in set(data.columns) - {kpi}]]
    target = data[kpi]
    return train_test_split(training, target, **options)


@tools.timeit
def main():
    # the first three lines do not contain meaningful data: they are dropped
    statistics = pd.read_csv(feature_constants.CSV_FILES["hammer_statistics"]).iloc[3:, :].set_index("timestamp")
    latency = statistics[["latency"]]

    network = pd.read_csv(feature_constants.CSV_FILES["net_interfaces"]).set_index("timestamp")
    ram = pd.read_csv(feature_constants.CSV_FILES["proc_stat_meminfo"]).set_index("timestamp")
    scheduler = pd.read_csv(feature_constants.CSV_FILES["proc_schedstat"]).set_index("timestamp")
    cpu = pd.read_csv(feature_constants.CSV_FILES["proc_stat_cpu"]).set_index("timestamp")

    data = latency.join([network, ram, scheduler, cpu]).dropna()
    columns = data[[col for col in set(data.columns) - {"latency"}]].columns

    train, test, target_train, target_test = prepare_data_for_kpi(data, "latency")

    model = setup_model(train, target_train, n_estimators=50)

    for item in sorted(zip(
            model.feature_importances_,
            columns
    ), reverse=True):
        print item

    import matplotlib.pyplot as plt
    from pandas.tools.plotting import scatter_matrix

    scatter_matrix(data.iloc[:500, :])
    plt.show()


    #print len(train)
    #print len(target_train)
    #print len(test)
    #print len(target_test)

    # model = setup_model()



##########################################################################################
if __name__ == '__main__':
    main()
