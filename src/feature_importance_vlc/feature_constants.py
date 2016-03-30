import os

DEBUG = True
N_ESTIMATORS = 5 if DEBUG else 100

DATA_FOLDER = "data"
CSV_FILES = {
    "hammer_statistics": os.path.join(DATA_FOLDER, "hammer_statistics-TOTAL.csv"),
    "net_interfaces": os.path.join(DATA_FOLDER, "net_interfaces.csv"),
    "proc_stat_cpu": os.path.join(DATA_FOLDER, "proc_stat_cpu.csv"),
    "proc_schedstat": os.path.join(DATA_FOLDER, "proc_schedstat.csv"),
    "proc_stat_meminfo": os.path.join(DATA_FOLDER, "proc_stat_meminfo.csv")
}
