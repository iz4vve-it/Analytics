"""
Constants module for feature importance in vlc kpi
"""

import os.path as path

DEBUG = True
N_ESTIMATORS = 5 if DEBUG else 1000
TREES_FEATURES_MODE = "log2"

SCORE_THRESHOLD = 0.1

DATA_FOLDER = path.join(path.abspath("."), "data")
DATA_FOLDER = DATA_FOLDER if "docs" not in DATA_FOLDER else \
    DATA_FOLDER.replace("docs/", "")

CSV_FILES = {
    "hammer_statistics": path.join(DATA_FOLDER, "hammer_statistics-TOTAL.csv"),
    "net_interfaces": path.join(DATA_FOLDER, "net_interfaces.csv"),
    "proc_stat_cpu": path.join(DATA_FOLDER, "proc_stat_cpu.csv"),
    "proc_schedstat": path.join(DATA_FOLDER, "proc_schedstat.csv"),
    "proc_stat_meminfo": path.join(DATA_FOLDER, "proc_stat_meminfo.csv")
}

CURRENT_KPI = 'latency'
