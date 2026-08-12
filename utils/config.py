from pathlib import Path

BASE_DIR = Path.cwd()

MAT_DATASETS = {
    2:  {"name": "Lung Discrete Dataset", "path": "datasets/lung_discrete.mat", "n_clusters": 7},
    3:  {"name": "COIL20 Dataset",        "path": "datasets/COIL20.mat",        "n_clusters": 20},
    8:  {"name": "Lymphoma Dataset",      "path": "datasets/lymphoma.mat",      "n_clusters": 9},
    26: {"name": "AR10P Dataset",         "path": "datasets/warpAR10P.mat",     "n_clusters": 10},
    27: {"name": "PIE10P Dataset",        "path": "datasets/warpPIE10P.mat",    "n_clusters": 10},
    28: {"name": "TOX-171 Dataset",       "path": "datasets/TOX_171.mat",       "n_clusters": 4},
}

CSV_DATASETS = {
    4:  {"name": "Heart Statlog Dataset", "path": "datasets/heart-statlog.dat", "sep": " ",
         "target_col": -1, "label_transform": "sub1", "drop_cols": [-1], "n_clusters": 2},

    6:  {"name": "Ionosphere Dataset", "path": "datasets/ionosphere.data", "sep": ",",
         "target_col": -1, "label_transform": {"g": 0, "else": 1}, "drop_cols": [-1], "n_clusters": 2},

    7:  {"name": "Liver Disorders Dataset", "path": "datasets/liver-disorders.data", "sep": ",",
         "target_col": -1, "label_transform": "sub1", "drop_cols": [-1], "n_clusters": 2},

    9:  {"name": "Lymphography Dataset", "path": "datasets/lymphography.data", "sep": ",",
         "target_col": 0, "label_transform": "sub1", "drop_cols": [0], "n_clusters": 4},

    10: {"name": "Monks Problem Dataset", "path": "datasets/monks-1.data", "sep": " ",
         "target_col": 0, "label_transform": "raw", "drop_cols": [0, -1], "n_clusters": 2},

    11: {"name": "Sonar Dataset", "path": "datasets/sonar.data", "sep": ",",
         "target_col": -1, "label_transform": {"R": 0, "else": 1}, "drop_cols": [-1], "n_clusters": 2},

    13: {"name": "Breast Cancer Wisconsin (Diagnostic) Dataset", "path": "datasets/wdbc.data", "sep": ",",
         "target_col": 1, "label_transform": {"M": 0, "else": 1}, "drop_cols": [0, 1], "n_clusters": 2},

    14: {"name": "Wine Dataset", "path": "datasets/wine.txt", "sep": ",",
         "target_col": 0, "label_transform": "raw", "drop_cols": [0], "n_clusters": 3},

    15: {"name": "Zoo Dataset", "path": "datasets/zoo.data", "sep": ",",
         "target_col": -1, "label_transform": "sub1", "drop_cols": [0, -1], "n_clusters": 7},

    16: {"name": "Iris Dataset", "path": "datasets/iris.txt", "sep": ",",
         "target_col": -1, "label_transform": "raw", "drop_cols": [-1], "n_clusters": 3},

    # POSSÍVEL BUG 4: path usa "AlgoritmosClusters/datasets/" em vez de "datasets/"
    17: {"name": "Glass Dataset", "path": "AlgoritmosClusters/datasets/glass.txt", "sep": ",",
         "target_col": -1, "label_transform": "raw", "drop_cols": [0, -1], "n_clusters": 6},

    # POSSÍVEL BUG 3: no original este path NÃO é combinado com current_dir
    18: {"name": "KC2 Dataset", "path": "datasets/kc2.txt", "sep": ",", "join_base": False,
         "target_col": -1, "label_transform": "raw", "drop_cols": [-1], "n_clusters": 2},
}

ARFF_DATASETS = {
    19: {"name": "Scene Dataset", "path": "datasets/scene.arff", "n_clusters": 2,
         "extra_drop_range": (294, 299)},
    20: {"name": "Madelon Dataset", "path": "datasets/madelon.arff", "n_clusters": 2},
    21: {"name": "Hiva Agnostic Dataset", "path": "datasets/hiva_agnostic.arff", "n_clusters": 2},
}

MUSK_CONFIG = {"name": "Musk (Version 1) Dataset", "path": "datasets/musk1.data", "n_clusters": 2}