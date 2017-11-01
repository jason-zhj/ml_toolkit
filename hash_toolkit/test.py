from hash_toolkit.metrics import get_mean_avg_precision

test_set = [
    {"hash":"001","label":1},
    {"hash":"100","label":0}
]

db_set = [
    {"hash":"001","label":1},
    {"hash":"001","label":1},
    {"hash":"001","label":0},
    {"hash":"010","label":0},
    {"hash":"100","label":1},
    {"hash":"100","label":1}
]

print(get_mean_avg_precision(test_set=test_set,db_set=db_set,maxdist=3))