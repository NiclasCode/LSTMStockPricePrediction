from typing import Tuple

import numpy as np
import pandas as pd
import pywt
import torch
from IPython.core.display_functions import display
from torch.utils.data import DataLoader

from classes.models.ExperimentConfig import ExperimentConfig
from classes.models.HyperparamConfig import HyperparamConfig
from classes.models.LoaderSet import LoaderSet
from classes.models.StockPredictionDataset import StockPredictionDataset


class DataManager:

    def __init__(self, exp_config: ExperimentConfig):

        self.exp_config = exp_config

        self.scaler = None
        self.df = None
        self.raw_data = None
        self.scaled_data = None

    def load_data(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.exp_config.file_path, header=[0, 1], index_col=0)
        self.df.columns = self.format_columns(self.df)
        self.raw_data = self.df.copy()
        display(self.raw_data)
        return self.df

    def select_features(self, features: list) -> pd.DataFrame:
        print("Features", features)
        print("Dataframe", self.df.columns.values)
        ordered = list(features)
        if self.exp_config.target in ordered:
            ordered.remove(self.exp_config.target)
        ordered.insert(0, self.exp_config.target)
        self.df = self.df[ordered]
        return self.df

    def denoise_series(self, x: np.ndarray) -> np.ndarray:
        coeff = pywt.wavedec(x, self.exp_config.wavelet)
        new_coeff = [coeff[0]]
        for detail in coeff[1:]:
            thr = self.exp_config.wavelet_thr_ratio * np.max(np.abs(detail))
            new_coeff.append(pywt.threshold(detail, value=thr, mode=self.exp_config.wavelet_mode))
        y = pywt.waverec(new_coeff, self.exp_config.wavelet)[0:len(x)]
        return y

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        self.scaler = self.exp_config.scaler_class()

        if self.exp_config.scale_mode == "global":
            scaled = self.scaler.fit_transform(df.values)
            return scaled

        if self.exp_config.scale_mode == "train_only":
            # fit only on train split, then transform all
            split_tt = int(len(df) * self.exp_config.train_test_ratio)
            split_tv = int(split_tt * self.exp_config.train_val_ratio)
            train_df = df.iloc[:split_tv]

            self.scaler.fit(train_df.values)
            scaled = self.scaler.transform(df.values)
            return scaled

        raise ValueError(f"Unknown scale_mode: {self.exp_config.scale_mode}")

    def prepare(self, features: list):
        self.load_data()
        self.select_features(features)

        if self.exp_config.drop_na:
            self.df.dropna(inplace=True)

        if self.exp_config.apply_wavelet:
            series = self.df[self.exp_config.target].to_numpy(dtype=float)
            y = self.denoise_series(series)
            self.df[self.exp_config.target] = pd.Series(y, index=self.df.index)

        if self.exp_config.scale:
            self.scaled_data = self.fit_transform(self.df)
        else:
            self.scaled_data = self.df.values

    @staticmethod
    def create_sequences(data, target, window_size, horizon=1):
        """
        data:      full feature array (n_steps, n_features)
        target:    1D array with the target (same length as data)
        window_size: length of input sequence
        horizon:   how many days ahead you want to predict (1 = next day,
                   5 = 5 days in the future, etc.)

        Returns:
            X: shape (N, window_size, n_features)
            y: shape (N,)
        """
        X, y = [], []

        # last index for i such that i + window_size - 1 + horizon < len(data)
        max_i = len(data) - window_size - horizon + 1

        for i in range(max_i):
            # sequence from i ... i+window_size-1
            X.append(data[i:i + window_size])
            # label at (last timestep + horizon)
            y.append(target[i + window_size - 1 + horizon])

        return np.array(X), np.array(y)

    def build_loaders(self, hyperparam_config: HyperparamConfig) -> LoaderSet:
        splits = self.split_data()

        loaders = []
        for split in splits:
            X, y = self.create_sequences(split, split[:, 0], hyperparam_config.window_size, hyperparam_config.horizon)
            X = torch.from_numpy(X).float()
            y = torch.from_numpy(y).float().unsqueeze(-1)
            ds = StockPredictionDataset(X, y)
            loader = DataLoader(ds,batch_size=hyperparam_config.batch_size,shuffle=False)
            loaders.append(loader)

        return LoaderSet(*loaders)

    def split_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        split data into train/val/test sets
        :return: data_train, data_val, data_trainval, data_test
        """
        assert self.scaled_data is not None
        split_tt = int(len(self.scaled_data) * self.exp_config.train_test_ratio)
        split_tv = int(split_tt * self.exp_config.train_val_ratio)

        data_train = self.scaled_data[:split_tv].copy()
        data_val = self.scaled_data[split_tv:split_tt].copy()
        data_test = self.scaled_data[split_tt:].copy()
        data_train_val = self.scaled_data[:split_tt].copy()
        return data_train, data_val, data_train_val, data_test

    def get_columns(self) -> list:
        """
        if df is already loaded, use it, otherwise load the first 5 lines from the file
        :return: list of columns
        """
        if self.df is not None and self.df.columns is not None:
            return self.df.columns.values
        return self.format_columns(pd.read_csv(self.exp_config.file_path, header=[0, 1], index_col=0, nrows=0))

    @staticmethod
    def format_columns(df) -> list:
        return ["_".join(col).strip() for col in df.columns.values]
