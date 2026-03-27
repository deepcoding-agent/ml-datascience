"""Model Registry — 27 algorithms for classification, regression, and clustering.

Each entry contains:
  - class_path: fully qualified sklearn/xgboost/lightgbm class
  - display_name: human-readable name
  - default_params: sensible defaults
  - tunable_params: Optuna search space (type, low, high) or (categorical, choices)
"""
from __future__ import annotations

MODEL_REGISTRY: dict[str, dict[str, dict]] = {
    # ── Classification (12) ──────────────────────────────────────────────────
    "classification": {
        "logistic_regression": {
            "class_path": "sklearn.linear_model.LogisticRegression",
            "display_name": "Logistic Regression",
            "default_params": {"max_iter": 1000, "random_state": 42},
            "tunable_params": {
                "C": ("float_log", 0.001, 100),
                "penalty": ("categorical", ["l1", "l2"]),
                "solver": ("categorical", ["liblinear", "saga"]),
            },
        },
        "decision_tree_clf": {
            "class_path": "sklearn.tree.DecisionTreeClassifier",
            "display_name": "Decision Tree",
            "default_params": {"random_state": 42},
            "tunable_params": {
                "max_depth": ("int", 2, 30),
                "min_samples_split": ("int", 2, 20),
                "min_samples_leaf": ("int", 1, 10),
            },
        },
        "random_forest_clf": {
            "class_path": "sklearn.ensemble.RandomForestClassifier",
            "display_name": "Random Forest",
            "default_params": {"n_estimators": 100, "random_state": 42},
            "tunable_params": {
                "n_estimators": ("int", 50, 500),
                "max_depth": ("int", 3, 30),
                "min_samples_split": ("int", 2, 20),
            },
        },
        "extra_trees_clf": {
            "class_path": "sklearn.ensemble.ExtraTreesClassifier",
            "display_name": "Extra Trees",
            "default_params": {"n_estimators": 100, "random_state": 42},
            "tunable_params": {
                "n_estimators": ("int", 50, 500),
                "max_depth": ("int", 3, 30),
                "min_samples_split": ("int", 2, 20),
            },
        },
        "gradient_boosting_clf": {
            "class_path": "sklearn.ensemble.GradientBoostingClassifier",
            "display_name": "Gradient Boosting",
            "default_params": {"n_estimators": 100, "random_state": 42},
            "tunable_params": {
                "n_estimators": ("int", 50, 300),
                "max_depth": ("int", 3, 10),
                "learning_rate": ("float_log", 0.01, 0.3),
                "subsample": ("float", 0.6, 1.0),
            },
        },
        "adaboost_clf": {
            "class_path": "sklearn.ensemble.AdaBoostClassifier",
            "display_name": "AdaBoost",
            "default_params": {"n_estimators": 50, "random_state": 42},
            "tunable_params": {
                "n_estimators": ("int", 30, 200),
                "learning_rate": ("float_log", 0.01, 2.0),
            },
        },
        "xgboost_clf": {
            "class_path": "xgboost.XGBClassifier",
            "display_name": "XGBoost",
            "default_params": {"n_estimators": 100, "random_state": 42, "eval_metric": "logloss", "verbosity": 0},
            "tunable_params": {
                "n_estimators": ("int", 50, 500),
                "max_depth": ("int", 3, 10),
                "learning_rate": ("float_log", 0.01, 0.3),
                "subsample": ("float", 0.6, 1.0),
                "colsample_bytree": ("float", 0.6, 1.0),
            },
        },
        "lightgbm_clf": {
            "class_path": "lightgbm.LGBMClassifier",
            "display_name": "LightGBM",
            "default_params": {"n_estimators": 100, "random_state": 42, "verbosity": -1},
            "tunable_params": {
                "n_estimators": ("int", 50, 500),
                "max_depth": ("int", 3, 15),
                "learning_rate": ("float_log", 0.01, 0.3),
                "num_leaves": ("int", 15, 127),
                "subsample": ("float", 0.6, 1.0),
            },
        },
        "svc_clf": {
            "class_path": "sklearn.svm.SVC",
            "display_name": "SVM (SVC)",
            "default_params": {"random_state": 42, "probability": True},
            "tunable_params": {
                "C": ("float_log", 0.01, 100),
                "kernel": ("categorical", ["rbf", "linear", "poly"]),
                "gamma": ("categorical", ["scale", "auto"]),
            },
        },
        "knn_clf": {
            "class_path": "sklearn.neighbors.KNeighborsClassifier",
            "display_name": "KNN",
            "default_params": {},
            "tunable_params": {
                "n_neighbors": ("int", 3, 25),
                "weights": ("categorical", ["uniform", "distance"]),
                "metric": ("categorical", ["euclidean", "manhattan"]),
            },
        },
        "naive_bayes_clf": {
            "class_path": "sklearn.naive_bayes.GaussianNB",
            "display_name": "Naive Bayes",
            "default_params": {},
            "tunable_params": {
                "var_smoothing": ("float_log", 1e-12, 1e-6),
            },
        },
        "mlp_clf": {
            "class_path": "sklearn.neural_network.MLPClassifier",
            "display_name": "MLP Neural Network",
            "default_params": {"max_iter": 500, "random_state": 42},
            "tunable_params": {
                "hidden_layer_sizes": ("categorical", [(64,), (128,), (64, 32), (128, 64)]),
                "learning_rate_init": ("float_log", 0.0001, 0.01),
                "alpha": ("float_log", 1e-5, 1e-2),
            },
        },
    },

    # ── Regression (11) ──────────────────────────────────────────────────────
    "regression": {
        "linear_regression": {
            "class_path": "sklearn.linear_model.LinearRegression",
            "display_name": "Linear Regression",
            "default_params": {},
            "tunable_params": {},
        },
        "ridge": {
            "class_path": "sklearn.linear_model.Ridge",
            "display_name": "Ridge",
            "default_params": {"random_state": 42},
            "tunable_params": {
                "alpha": ("float_log", 0.001, 100),
            },
        },
        "lasso": {
            "class_path": "sklearn.linear_model.Lasso",
            "display_name": "Lasso",
            "default_params": {"random_state": 42, "max_iter": 5000},
            "tunable_params": {
                "alpha": ("float_log", 0.0001, 10),
            },
        },
        "elastic_net": {
            "class_path": "sklearn.linear_model.ElasticNet",
            "display_name": "ElasticNet",
            "default_params": {"random_state": 42, "max_iter": 5000},
            "tunable_params": {
                "alpha": ("float_log", 0.0001, 10),
                "l1_ratio": ("float", 0.1, 0.9),
            },
        },
        "decision_tree_reg": {
            "class_path": "sklearn.tree.DecisionTreeRegressor",
            "display_name": "Decision Tree",
            "default_params": {"random_state": 42},
            "tunable_params": {
                "max_depth": ("int", 2, 30),
                "min_samples_split": ("int", 2, 20),
            },
        },
        "random_forest_reg": {
            "class_path": "sklearn.ensemble.RandomForestRegressor",
            "display_name": "Random Forest",
            "default_params": {"n_estimators": 100, "random_state": 42},
            "tunable_params": {
                "n_estimators": ("int", 50, 500),
                "max_depth": ("int", 3, 30),
                "min_samples_split": ("int", 2, 20),
            },
        },
        "gradient_boosting_reg": {
            "class_path": "sklearn.ensemble.GradientBoostingRegressor",
            "display_name": "Gradient Boosting",
            "default_params": {"n_estimators": 100, "random_state": 42},
            "tunable_params": {
                "n_estimators": ("int", 50, 300),
                "max_depth": ("int", 3, 10),
                "learning_rate": ("float_log", 0.01, 0.3),
                "subsample": ("float", 0.6, 1.0),
            },
        },
        "xgboost_reg": {
            "class_path": "xgboost.XGBRegressor",
            "display_name": "XGBoost",
            "default_params": {"n_estimators": 100, "random_state": 42, "verbosity": 0},
            "tunable_params": {
                "n_estimators": ("int", 50, 500),
                "max_depth": ("int", 3, 10),
                "learning_rate": ("float_log", 0.01, 0.3),
                "subsample": ("float", 0.6, 1.0),
                "colsample_bytree": ("float", 0.6, 1.0),
            },
        },
        "lightgbm_reg": {
            "class_path": "lightgbm.LGBMRegressor",
            "display_name": "LightGBM",
            "default_params": {"n_estimators": 100, "random_state": 42, "verbosity": -1},
            "tunable_params": {
                "n_estimators": ("int", 50, 500),
                "max_depth": ("int", 3, 15),
                "learning_rate": ("float_log", 0.01, 0.3),
                "num_leaves": ("int", 15, 127),
            },
        },
        "svr_reg": {
            "class_path": "sklearn.svm.SVR",
            "display_name": "SVR",
            "default_params": {},
            "tunable_params": {
                "C": ("float_log", 0.01, 100),
                "kernel": ("categorical", ["rbf", "linear", "poly"]),
                "gamma": ("categorical", ["scale", "auto"]),
            },
        },
        "knn_reg": {
            "class_path": "sklearn.neighbors.KNeighborsRegressor",
            "display_name": "KNN",
            "default_params": {},
            "tunable_params": {
                "n_neighbors": ("int", 3, 25),
                "weights": ("categorical", ["uniform", "distance"]),
            },
        },
    },

    # ── Clustering (4) ───────────────────────────────────────────────────────
    "clustering": {
        "kmeans": {
            "class_path": "sklearn.cluster.KMeans",
            "display_name": "KMeans",
            "default_params": {"n_init": 10, "random_state": 42},
            "tunable_params": {
                "n_clusters": ("int", 2, 15),
            },
        },
        "dbscan": {
            "class_path": "sklearn.cluster.DBSCAN",
            "display_name": "DBSCAN",
            "default_params": {},
            "tunable_params": {
                "eps": ("float", 0.1, 5.0),
                "min_samples": ("int", 3, 20),
            },
        },
        "hierarchical": {
            "class_path": "sklearn.cluster.AgglomerativeClustering",
            "display_name": "Hierarchical",
            "default_params": {},
            "tunable_params": {
                "n_clusters": ("int", 2, 15),
                "linkage": ("categorical", ["ward", "complete", "average"]),
            },
        },
        "gaussian_mixture": {
            "class_path": "sklearn.mixture.GaussianMixture",
            "display_name": "Gaussian Mixture",
            "default_params": {"random_state": 42},
            "tunable_params": {
                "n_components": ("int", 2, 15),
                "covariance_type": ("categorical", ["full", "tied", "diag", "spherical"]),
            },
        },
    },
}


def instantiate_model(task_type: str, algo_key: str, params: dict | None = None) -> object:
    """Create a model instance from registry entry, overriding defaults with params."""
    entry = MODEL_REGISTRY[task_type][algo_key]
    class_path = entry["class_path"]
    module_path, class_name = class_path.rsplit(".", 1)

    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    merged = {**entry["default_params"], **(params or {})}
    return cls(**merged)


def get_algorithms_for_task(task_type: str) -> dict[str, dict]:
    """Return all registered algorithms for a given task type."""
    return MODEL_REGISTRY.get(task_type, {})


def select_algorithms(
    task_type: str,
    n_rows: int,
    n_features: int,
    n_classes: int | None = None,
) -> list[str]:
    """Smart algorithm selection based on dataset characteristics.

    Returns 5-8 algorithm keys suited to the data.
    """
    all_algos = list(MODEL_REGISTRY.get(task_type, {}).keys())
    if task_type == "clustering":
        return all_algos  # only 4, use all

    selected: list[str] = []

    if task_type == "classification":
        # Always include
        selected.append("logistic_regression")
        selected.append("random_forest_clf")
        selected.append("xgboost_clf")

        if n_rows > 5000:
            selected.append("lightgbm_clf")
        if n_rows < 5000:
            selected.append("svc_clf")
            selected.append("knn_clf")
        if n_features > 50:
            selected.append("extra_trees_clf")
        selected.append("gradient_boosting_clf")
        if n_rows < 10000:
            selected.append("adaboost_clf")

    elif task_type == "regression":
        # Always include strong baselines + boosting
        selected.append("random_forest_reg")
        selected.append("xgboost_reg")
        selected.append("lightgbm_reg")
        selected.append("gradient_boosting_reg")
        selected.append("ridge")
        selected.append("linear_regression")
        if n_rows < 5000:
            selected.append("svr_reg")
            selected.append("knn_reg")
        if n_features > 20:
            selected.append("lasso")
            selected.append("elastic_net")

    # Deduplicate while preserving order, cap at 8
    seen: set[str] = set()
    unique = []
    for a in selected:
        if a not in seen and a in all_algos:
            seen.add(a)
            unique.append(a)
    return unique[:8]
