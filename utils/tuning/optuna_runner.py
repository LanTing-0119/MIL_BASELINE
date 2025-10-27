"""Optuna integration for MIL baseline training."""

from __future__ import annotations

import glob
import os
from copy import deepcopy
from typing import Any, Dict, Iterable, Optional

import optuna
import pandas as pd

from process.process_all import process
from utils.general_utils import get_time


class OptunaTuner:
    """Run Optuna based hyper-parameter optimisation for MIL experiments."""

    def __init__(self, base_args, yaml_path: str, cli_options: Optional[Iterable[str]] = None) -> None:
        self._base_args = base_args
        self._yaml_path = yaml_path
        self._cli_options = cli_options
        self._tuning_cfg = base_args.Tuning
        self._study_name = getattr(self._tuning_cfg, "study_name", None) or f"study_{get_time()}"
        self._direction = getattr(self._tuning_cfg, "direction", "maximize")
        self._metric = getattr(self._tuning_cfg, "metric", "val_macro_auc")
        self._n_trials = int(getattr(self._tuning_cfg, "n_trials", 10))
        self._sub_dir = None

        if self._direction not in ("maximize", "minimize"):
            raise ValueError("Tuning.direction must be either 'maximize' or 'minimize'.")

        if self._base_args.Dataset.dataset_root_dir not in ({}, None):
            raise ValueError(
                "Optuna tuning currently expects a single CSV split (Dataset.dataset_csv_path). "
                "Please disable dataset_root_dir or extend the tuner to handle k-fold splits."
            )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def optimize(self) -> optuna.study.Study:
        sampler = self._build_sampler(getattr(self._tuning_cfg, "sampler", None))
        pruner = self._build_pruner(getattr(self._tuning_cfg, "pruner", None))
        study = optuna.create_study(direction=self._direction, sampler=sampler, pruner=pruner, study_name=self._study_name)
        study.optimize(self._objective, n_trials=self._n_trials)
        self._save_study_results(study)
        self._log_best_trial(study)
        return study

    # ------------------------------------------------------------------
    # optuna helpers
    # ------------------------------------------------------------------
    def _objective(self, trial: optuna.trial.Trial) -> float:
        args = deepcopy(self._base_args)
        self._apply_trial_params(trial, args)
        self._prepare_log_dir(args, trial.number)
        process(args, self._yaml_path, self._cli_options)
        return self._extract_metric(args)

    def _apply_trial_params(self, trial: optuna.trial.Trial, args) -> None:
        search_space = getattr(self._tuning_cfg, "search_space", None)
        if not search_space:
            return
        for dotted_key, spec in search_space.items():
            value = self._suggest_value(trial, dotted_key, spec)
            self._assign_nested_value(args, dotted_key, value)

    def _suggest_value(self, trial: optuna.trial.Trial, dotted_key: str, spec: Any) -> Any:
        param_type = getattr(spec, "type", None)
        if param_type is None:
            raise ValueError(f"Missing 'type' for search space entry '{dotted_key}'.")
        param_type = str(param_type).lower()

        if param_type in {"float", "uniform"}:
            return trial.suggest_float(
                dotted_key,
                float(spec.low),
                float(spec.high),
                step=self._maybe_float(spec, "step"),
            )
        if param_type in {"logfloat", "loguniform"}:
            return trial.suggest_float(
                dotted_key,
                float(spec.low),
                float(spec.high),
                log=True,
            )
        if param_type == "int":
            step = int(getattr(spec, "step", 1))
            return trial.suggest_int(dotted_key, int(spec.low), int(spec.high), step=step)
        if param_type == "categorical":
            choices = list(spec.choices)
            if not choices:
                raise ValueError(f"Search space entry '{dotted_key}' must define non-empty 'choices'.")
            return trial.suggest_categorical(dotted_key, choices)
        raise ValueError(f"Unsupported search space type '{param_type}' for '{dotted_key}'.")

    def _assign_nested_value(self, args, dotted_key: str, value: Any) -> None:
        keys = dotted_key.split('.')
        target = args
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value

    def _prepare_log_dir(self, args, trial_number: int) -> None:
        log_root_dir = args.Logs.log_root_dir
        os.makedirs(log_root_dir, exist_ok=True)
        self._sub_dir = os.path.join(log_root_dir, args.Dataset.DATASET_NAME, args.General.MODEL_NAME, "tuning", self._study_name)
        os.makedirs(self._sub_dir, exist_ok=True)
        trial_dir = os.path.join(
            self._sub_dir,
            f'trial_{trial_number:03d}_time_{get_time()}'
        )
        os.makedirs(trial_dir, exist_ok=True)
        args.Logs.now_log_dir = trial_dir

    def _extract_metric(self, args) -> float:
        best_log = self._locate_best_log(args.Logs.now_log_dir)
        df = pd.read_csv(best_log)
        if self._metric not in df.columns:
            raise KeyError(
                f"Metric '{self._metric}' not found in {best_log}. Available columns: {list(df.columns)}"
            )
        return float(df.loc[df.index[0], self._metric])

    def _locate_best_log(self, log_dir: str) -> str:
        pattern = os.path.join(log_dir, 'Best_Log_seed*.csv')
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
        global_pattern = os.path.join(log_dir, 'Log_seed*.csv')
        matches = glob.glob(global_pattern)
        if not matches:
            raise FileNotFoundError(
                f"No log files matching 'Best_Log_seed*.csv' or 'Log_seed*.csv' found in {log_dir}."
            )
        return matches[0]

    def _save_study_results(self, study: optuna.study.Study) -> None:
        if self._sub_dir is None:
            return
        study_df = study.trials_dataframe()
        study_csv = os.path.join(self._sub_dir, f'{self._study_name}_trials.csv')
        study_df.to_csv(study_csv, index=False)

    def _log_best_trial(self, study: optuna.study.Study) -> None:
        best_trial = study.best_trial
        print("Best trial:")
        print(f"  Value: {best_trial.value}")
        print("  Params:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")

    def _build_sampler(self, sampler_cfg: Optional[Dict[str, Any]]):
        if not sampler_cfg:
            return None
        name = str(getattr(sampler_cfg, "name", "tpe")).lower()
        seed = getattr(sampler_cfg, "seed", None)
        if name == "tpe":
            return optuna.samplers.TPESampler(seed=seed)
        if name == "random":
            return optuna.samplers.RandomSampler(seed=seed)
        if name == "grid":
            search_space = getattr(self._tuning_cfg, "search_space", None)
            if not search_space:
                raise ValueError("GridSampler requires a defined search_space.")
            grid = {key: list(spec.choices) for key, spec in search_space.items() if getattr(spec, "type", "").lower() == "categorical"}
            return optuna.samplers.GridSampler(grid)
        raise ValueError(f"Unsupported sampler '{name}'.")

    def _build_pruner(self, pruner_cfg: Optional[Dict[str, Any]]):
        if not pruner_cfg:
            return None
        name = str(getattr(pruner_cfg, "name", "nopruner")).lower()
        if name == "nopruner":
            return optuna.pruners.NopPruner()
        if name == "median":
            n_warmup_steps = int(getattr(pruner_cfg, "warmup_steps", 0))
            return optuna.pruners.MedianPruner(n_warmup_steps=n_warmup_steps)
        if name == "successivehalving":
            return optuna.pruners.SuccessiveHalvingPruner()
        if name == "hyperband":
            return optuna.pruners.HyperbandPruner()
        raise ValueError(f"Unsupported pruner '{name}'.")

    @staticmethod
    def _maybe_float(spec: Any, key: str) -> Optional[float]:
        if hasattr(spec, key):
            return float(getattr(spec, key))
        return None
