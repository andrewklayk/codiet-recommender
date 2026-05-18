import ast
import time
import zipfile
from os.path import join
import logging
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig

from data_helper import load_all_data
from experiments_utils import log_system_info, log_params_from_omegaconf_dict
from recommender import run_recommender
from recommender_utils import run_feature_selection_scikit
from utils import log_exceptions, plot_heatmap

logger = logging.getLogger(__name__)

import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
from mlflow import log_metric, log_metrics, log_param, log_artifact, log_text, log_table, log_dict


@hydra.main(version_base=None,  config_path="./experiments_conf", config_name="config")
@log_exceptions
def start_experiment(cfg: DictConfig) -> None:
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    mlflow.set_experiment(cfg.experiment) #mlflow.set_experiment(experiment_name=cfg.experiment)
    conf_yaml = OmegaConf.to_yaml(cfg)
    print(conf_yaml)
    app_config = OmegaConf.to_container(cfg)
    with mlflow.start_run() as mlrun:
        logger.info(f'Starting experiment in {output_dir}')
        log_text(output_dir, 'work_dir.txt')
        with open(join(output_dir, 'config.yaml'), 'w') as f:
            f.write(conf_yaml)
        log_artifact(join(output_dir, 'config.yaml'))
        log_params_from_omegaconf_dict(cfg)
        log_system_info(HydraConfig.get())

        prep_data = None
        if cfg.problem.name == "codiet":
            food_feats, non_food_feats, prep_data = load_all_data()
            with zipfile.ZipFile("./data/W_est.csv.zip") as z:
                with z.open("W_est.csv") as f:
                    w_est = np.loadtxt(f, delimiter=",")
                # w_est = np.loadtxt("./data/W_est.csv", delimiter=",")
            mlflow.log_artifact("./data/W_est.csv.zip")
            s = Path('./data/intra_nodes.txt').read_text(encoding="utf-8").strip()
            mlflow.log_text(s, 'intra_nodes.txt')
            row_and_col_names = ast.literal_eval(s) #[x.strip() for x in s.strip("[]").split(",")]

        if cfg.problem.name == "codiet-select":
            prep_data = pd.read_feather(join(cfg.problem.data_path, "features.feather"))
            prep_data = prep_data.select_dtypes(include=["number"])
            row_and_col_names = prep_data.columns
            with zipfile.ZipFile(join(cfg.problem.data_path, "W_est.csv.zip")) as z:
                with z.open(f"W_est_{cfg.problem.regularization}.csv") as f:
                    df = pd.read_csv(f, index_col=0, header=0)
                    # test that both data frames have the same index!
                    pd.testing.assert_index_equal(prep_data.columns, df.columns)
                    w_est = df.to_numpy()

        if cfg.problem.name == "cds":
            import cds_utils
            prep_data = cds_utils.load_data(cfg.problem.n, cfg.problem.granularity, cfg.problem.p, cfg.problem.data_path)
        if cfg.problem.name == "industry_eu":
            from data_industry import load_data

            prep_data = load_data(
                cfg.problem.data_path,
                frequency=cfg.problem.get("frequency"),
                target=cfg.problem.get("target"),
                features=cfg.problem.get("features"),
                start_date=cfg.problem.get("start_date"),
                end_date=cfg.problem.get("end_date"),
                impute=cfg.problem.get("impute", "none"),
                dropna_selected=cfg.problem.get("dropna_selected", True),
            )
        if cfg.problem.name == 'Sachs':
            import sachs_utils
            #from sachs_utils import load_data
            prep_data = sachs_utils.load_data(cfg.problem.variant, cfg.problem.normalize, cfg.problem.data_path)
        if cfg.problem.name in ["cds", "Sachs"]:
            with zipfile.ZipFile(join(cfg.problem.data_path, "W_est.csv.zip")) as z:
                with z.open(f"W_est_{cfg.problem.name}.csv") as f:
                    w_est = np.loadtxt(f, delimiter=",")
            mlflow.log_artifact(join(cfg.problem.data_path, "W_est.csv.zip"))
            row_and_col_names = prep_data.columns
        if cfg.problem.name == "industry_eu":
            row_and_col_names = prep_data.columns
            w_est = np.zeros((len(row_and_col_names), len(row_and_col_names)))

        print(row_and_col_names)
        start_time = time.time()
        target_feat = cfg.problem.target
        full_feats = cfg.problem.features
        if 'N_SELECT_FEATURES' in cfg.solver:
            n_select_features = cfg.solver.N_SELECT_FEATURES
        else:
            n_select_features = len(full_feats)
        current_column_names = list(prep_data.columns)  # food_feats + non_food_feats
        mlflow.log_text('[' + ", ".join(current_column_names) + "]", "data_feats_list.txt")
        curr_feats, curr_train_errs, curr_test_errs, all_train_errs, all_train_cs, all_test_errs, all_test_cs, w_est = run_feature_selection_scikit(
            prep_data,
            cfg.solver.model_name,
            cfg.solver.custom_objective,
            target_feat,
            w_est, row_and_col_names,
            cfg.solver.n_runs, n_select_features,
            full_feats,
            solver_cfg=cfg.solver,
            model_factory=None,

        )

        if w_est is not None:
            np.savetxt(join(output_dir, 'W_est.csv'), w_est, delimiter=',')
            mlflow.log_artifact(join(output_dir, 'W_est.csv'))
            if full_feats is not None:
                intra_nodes = full_feats + [target_feat]
            else:
                intra_nodes = list(prep_data.columns) + [target_feat]
            plot_heatmap(w_est, intra_nodes, intra_nodes, filename=join(output_dir, f'W_est_heatmap.png'))
            mlflow.log_artifact(join(output_dir, f'W_est_heatmap.png'))

        # res_dict[target_col] = (curr_feats, curr_train_errs, curr_test_errs)

        #return curr_feats, curr_train_errs, curr_test_errs, all_train_errs, all_test_errs


        #curr_feats, curr_train_err, curr_test_err, all_train_errs, all_test_errs = run_recommender(food_feats, non_food_feats, prep_data, target_feat, w_est, row_and_col_names, cfg.solver.model_name, cfg.solver.custom_objective, cfg.solver.N_SELECT_FEATURES, cfg.solver.n_runs, cfg.solver)

        log_dict({
            'train_errs': all_train_errs.tolist(),
            'train_constr': all_train_cs.tolist(),
            'test_errs': all_test_errs.tolist(),
            'test_cs': all_test_cs.tolist()},
            'cv_errors.yaml')
        # print(result)
        #
        # assert len(result) == 1

        #for target_feat, (curr_feats, curr_train_errs, curr_test_errs) in result.items():

        target_feat = target_feat.replace(' ', '_').replace('(', '_').replace(')', '_').replace('/', '_')
        mlflow.log_text('['+", ".join(curr_feats) + "]", "selected_features.txt")

        # train_err = curr_train_errs[-1]
        # test_err = curr_test_errs[-1]
        log_metric(f'train_error', curr_train_errs)
        log_metric(f'test_err', curr_test_errs)


        solving_duration = time.time() - start_time

        log_metric('runtime', solving_duration)

        logger.info(f'Experiment Finished')


if __name__ == "__main__":
    start_experiment()
