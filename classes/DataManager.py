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
        """
        Loads data into a pandas DataFrame. If raw data is available, it returns a copy of the
        raw data. Otherwise, it reads the data from the file path specified in the configuration
        and applies necessary column formatting.

        :raises FileNotFoundError: Raised when the file specified in the configuration does not exist.
        :raises pd.errors.ParserError: Raised if there is an issue parsing the CSV file.

        :return: A pandas DataFrame containing the loaded and optionally transformed data.
        :rtype: pd.DataFrame
        """
        # if already loaded, return a copy
        if self.raw_data is not None:
            self.df = self.raw_data.copy()
            return self.df

        # if not loaded yet, load from file and save in raw, if needed again later
        self.df = pd.read_csv(self.exp_config.file_path, header=[0, 1], index_col=0)
        self.df.columns = self.format_columns(self.df)
        self.raw_data = self.df.copy()
        return self.df

    def select_features(self, features: list) -> pd.DataFrame:
        """
        Selects and reorders features in the DataFrame based on the given list of feature names.
        The target feature, as specified in the configuration, is moved to the front.

        :param features: A list of feature names to reorder and select from the DataFrame.
        :type features: list
        :return: A pandas DataFrame with the reordered and filtered features.
        :rtype: pd.DataFrame
        """
        ordered = list(features)
        if self.exp_config.target in ordered:
            ordered.remove(self.exp_config.target)
        ordered.insert(0, self.exp_config.target)
        self.df = self.df[ordered]
        return self.df

    def denoise_series(self, x: np.ndarray) -> np.ndarray:
        """
        Apply wavelet denoising to the given time series.
        :param x: time series
        :return: denoised time series
        """
        coeff = pywt.wavedec(x, self.exp_config.wavelet)
        new_coeff = [coeff[0]]
        for detail in coeff[1:]:
            thr = self.exp_config.wavelet_thr_ratio * np.max(np.abs(detail))
            new_coeff.append(pywt.threshold(detail, value=thr, mode=self.exp_config.wavelet_mode))
        y = pywt.waverec(new_coeff, self.exp_config.wavelet)[0:len(x)]
        return y

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Scales the input DataFrame based on the scaling configuration provided. This method fits the
        scaler to the data and transforms it.

        :param df: Pandas DataFrame containing the data to be scaled.
        :return: Scaled data as a NumPy array.
        """

        self.scaler = self.exp_config.scaler_class()
        scaled = self.scaler.fit_transform(df.values)
        return scaled

    def prepare(self, features: list):
        """
        Prepares the dataset by performing a series of preprocessing steps such as loading
        data, selecting features, handling missing values, applying a wavelet denoising
        procedure, and scaling the data.

        :param features: A list of features/columns to be selected in the dataset
        """
        self.load_data()
        self.select_features(features)

        if self.exp_config.drop_na:
            self.df.dropna(inplace=True)

        if self.exp_config.apply_wavelet:
            series = self.df[self.exp_config.target].to_numpy(dtype=float)
            y = self.denoise_series(series)
            self.df[self.exp_config.target] = pd.Series(y, index=self.df.index)

        self.scaled_data = self.fit_transform(self.df)

    def build_loaders(self, hyperparam_config: HyperparamConfig) -> LoaderSet:
        """
        Builds and returns data loaders for each data split, tailored for stock prediction tasks.
        This method processes the data into sequences based on the specified hyperparameters,
        converts them to PyTorch tensors, and wraps them into DataLoader objects.

        :param hyperparam_config: Configuration object containing hyperparameters for data
            processing and loader creation, including window size, horizon, and batch size.
        :type hyperparam_config: HyperparamConfig
        :return: A set of data loaders for different data splits.
        :rtype: LoaderSet
        """
        splits = self.split_data()

        loaders = []
        for split in splits:
            X, y = self.create_sequences(split, split[:, 0], hyperparam_config.window_size, hyperparam_config.horizon)
            X = torch.from_numpy(X).float()
            y = torch.from_numpy(y).float().unsqueeze(-1)
            ds = StockPredictionDataset(X, y)
            loader = DataLoader(ds, batch_size=hyperparam_config.batch_size, shuffle=False)
            loaders.append(loader)

        return LoaderSet(*loaders)

    def split_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        split data into train/val/test sets, data has to be scaled beforehand using prepare()
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
        Retrieve and format the columns from a specified file.

        This method reads the header of a file configured in the experiment configuration
        and retrieves its columns. The columns are formatted and returned as a list.

        :return: A list of formatted column names extracted from the file header.
        """
        return self.format_columns(pd.read_csv(self.exp_config.file_path, header=[0, 1], index_col=0, nrows=0))

    @staticmethod
    def create_sequences(data, target, window_size, horizon=1):
        """
        Create sequences from given time-series data for supervised learning purposes. This
        method generates feature sequences (X) and corresponding target values (y) based on the
        specified window size and horizon. The resulting dataset can then be used for training
        machine learning models for time-series prediction.

        :param data: Array-like time-series input data to create sequences from.
        :param target: Array-like target values corresponding to the input data.
        :param window_size: Number of time steps to include in each generated sequence.
        :param horizon: Lead time for the prediction, indicating how many steps into the future
                        the target should align with. Default is 1.
        :return: A tuple containing feature sequences (X) and corresponding target values (y),
                 both as NumPy arrays.
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

    @staticmethod
    def format_columns(df) -> list:
        return ["_".join(col).strip() for col in df.columns.values]
