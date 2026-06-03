import mlflow
import numpy as np
import logging
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate
import torch


from compute_tools import compute_predictor_errors
from recommender_estimator import XGBRecommenderPredictor, REGRecommenderPredictor, HCRecommenderPredictor, compute_predictor_errors_and_cs_scikit, compute_predictor_errors_scikit
from discrete_estimator import DiscreteRecommenderPredictor, compute_discrete_predictor_errors_scikit


def get_mean_average_errors(prep_data, run_feats, target_col, w_est, row_and_col_names,
                            model_name, custom_objective,
                            n_runs):
    run_res = []
    for i in range(n_runs):
        run_res.append(
            compute_predictor_errors(prep_data, run_feats, target_col, w_est, row_and_col_names,
                                     model_name=model_name,
                                     custom_objective=custom_objective,
                                     do_print=False, stack_linear=False,
                                     compute_covs=False
                                     )
        )

    train_ratio_mean = np.array([v[4] for v in run_res]).mean()
    test_ratio_mean = np.array([v[2] for v in run_res]).mean()

    return train_ratio_mean, test_ratio_mean


from time import time


def run_feature_selection(prep_data, model_name,custom_objective,
                          target_col,
                          w_est, row_and_col_names,
                          n_runs, n_features,
                          full_feats,
                          model_factory=None
                          ):
    curr_feats = []

    # these are actually variance ratios, not pure errors.
    curr_test_errs = []
    curr_train_errs = []

    curr_test_error = 100000.

    print(f'====== {target_col} ')
    for i in range(n_features):

        if model_factory is not None:
            model_class, model_params = model_factory(len(curr_feats) + 1)

        start_time = time()

        candidate_feats = [s for s in full_feats if s not in curr_feats]

        res = []
        for feat in candidate_feats:
            train_ratio_mean, test_ratio_mean = get_mean_average_errors(
                prep_data, curr_feats + [feat], target_col,
                w_est, row_and_col_names,
                model_name,custom_objective,
                n_runs
            )

            res.append((feat, train_ratio_mean, test_ratio_mean))

        res = sorted(res, key=lambda t: t[2])

        best_res = res[0]

        best_feat = best_res[0]
        best_test_error = best_res[2]
        best_train_error = best_res[1]

        print('')
        print(f'Round {i}')

        if best_test_error >= curr_test_error:
            print('Failed to impove further')
            break

        curr_feats += [best_feat]
        curr_test_errs += [best_test_error]
        curr_train_errs += [best_train_error]

        curr_test_error = best_test_error

        print(f'   Features: {curr_feats}')
        print(f'   Mean Train var reduction: {100 * (1. - best_res[1])}')
        print(f'   Mean Test var reduction: {100 * (1. - best_test_error)}')

        ctime = time()
        print(f'   Round completed in {(ctime - start_time) / 60:.2f} min.')

    return curr_feats, curr_train_errs, curr_test_errs


def create_model(model_name, w_est, target_col, row_and_col_names, custom_objective, prep_data,
                                        solver_cfg):
    model = None
    if model_name == "XGB":
        model = XGBRecommenderPredictor(w_est, target_col, row_and_col_names, custom_objective, prep_data,
                                        solver_cfg)  # prep_data[row_and_col_names]
    elif model_name == "REG":
        model = REGRecommenderPredictor(w_est, target_col, row_and_col_names, custom_objective, prep_data, solver_cfg)
    elif model_name == "HC":
        torch.manual_seed(42)
        model = HCRecommenderPredictor(w_est, target_col, row_and_col_names, custom_objective,
                                       prep_data, solver_cfg)
    elif model_name == "DISC":
        model = DiscreteRecommenderPredictor(w_est, target_col, row_and_col_names, custom_objective,
                                             prep_data, solver_cfg)
    if model is None:
        raise ValueError("Model can be only XGB, REG, HC, or DISC.")
    return model


def run_feature_selection_scikit(prep_data, model_name, custom_objective,
                          target_col,
                          w_est, row_and_col_names,
                          n_runs, n_features,
                          full_feats, solver_cfg, model_factory=None
                          ):
    logging.info(f"w_est shape {w_est.shape}")
    logging.info(f"target_col {target_col}")
    logging.info(f"row_and_col_names {len(row_and_col_names)}")
    logging.info(f"prep_data.shape {prep_data.shape}")
    logging.info(f"row_and_col_names {row_and_col_names}")



    model = create_model(model_name, w_est, target_col, row_and_col_names, custom_objective, prep_data,
                                        solver_cfg)
    #mlflow.log_param("feature_selector", "SequentialFeatureSelector")


    assert all(isinstance(col, str) for col in prep_data.columns)
    prep_data = prep_data.dropna(subset=[target_col])
    y = prep_data[target_col]

    if isinstance(model, DiscreteRecommenderPredictor):
        counts = np.bincount(y.astype(int))
        score_normalizer = max(1e-9, 1.0 - counts.max() / len(y))
    else:
        score_normalizer = mean_squared_error(y, np.ones_like(y) * y.mean())
    #TODO this deletes only 30 columns out of 508, so I will do this so that scikit feature selection works (no nas)
    #prep_data = prep_data[full_feats]
    prep_data = prep_data.dropna(axis=1)
    X = prep_data.drop(target_col, axis=1) if target_col in prep_data.columns else prep_data
    if full_feats is not None:
        X = X[full_feats]

    logging.info(f"Testing on columns {len(X.columns)}: {X.columns}")

    _scorer = compute_predictor_errors_scikit if isinstance(model, DiscreteRecommenderPredictor) else compute_predictor_errors_scikit

    if 'feature_selector' in solver_cfg and solver_cfg.feature_selector == 'SequentialFeatureSelector':
        sfs = SequentialFeatureSelector(
            model,
            direction="forward",
            scoring=_scorer,
            cv=n_runs,
            n_features_to_select=n_features
        )

        sfs.fit(X, y)
        # here are the selected features
        best_features = X.columns[sfs.get_support()]
        mlflow.log_metric("number_of_best_features", len(best_features))
        mlflow.log_dict({'selected_best_features': list(best_features)}, "selected_features.yaml")
        selected_indices = sfs.get_support(indices=True)

        logging.info(f"Best features {best_features}, {selected_indices}")

        X_selected = X.iloc[:, selected_indices]
        X_selected = X_selected.to_numpy() # CV did not work with pandas data frame, IDKW
        w_est = model._w_est
        
    else:
        model.fit(X, y)
        X_selected = X.to_numpy()
        best_features = full_feats
        w_est = model._w_est

    results = cross_validate(model, X_selected, y,
        cv=n_runs,
        scoring=(lambda estimator, X, y: compute_predictor_errors_and_cs_scikit(estimator, X, y, estimator._w_est)) if isinstance(model, HCRecommenderPredictor) else _scorer,
        return_train_score=True
    )

    if model_name == 'HC':
        # add fit call to get the graph for selected features
        model.fit(X_selected, y)
        w_est = model._w_est
        torch.save(model._rf_model_.state_dict(), "model.pt")
    test_mse_fold = results["test_score"]
    test_mse = test_mse_fold.mean() / score_normalizer
    train_mse_fold = results["train_score"]
    train_mse = train_mse_fold.mean() / score_normalizer

    if 'train_c' not in results:
        results['train_c'] = np.zeros_like(test_mse_fold)
        results['test_c'] = np.zeros_like(test_mse_fold)

    return best_features, train_mse, test_mse, train_mse_fold / score_normalizer, results['train_c'], test_mse_fold / score_normalizer, results["test_c"], w_est