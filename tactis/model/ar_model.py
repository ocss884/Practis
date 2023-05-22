"""
Copyright 2022 ServiceNow
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

>> The highest level model for TACTiS, which contains both its encoder and decoder.
"""


from typing import Any, Dict, Optional, Tuple

import numpy
import torch
from torch import nn


class NormalizationIdentity:
    """
    Trivial normalization helper. Do nothing to its data.
    """

    def __init__(self, hist_value: torch.Tensor):
        """
        Parameters:
        -----------
        hist_value: torch.Tensor [batch, series, time steps]
            Historical data which can be used in the normalization.
        """
        std, mean = torch.std_mean(hist_value, dim=2, unbiased=True, keepdim=True)
        self.std = torch.ones_like(std)#std.clamp(min=1e-8)#
        self.mean = mean
        pass

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        """
        Normalize the given values according to the historical data sent in the constructor.

        Parameters:
        -----------
        value: Tensor [batch, series, time steps]
            A tensor containing the values to be normalized.

        Returns:
        --------
        norm_value: Tensor [batch, series, time steps]
            The normalized values.
        """
        return value

    def denormalize(self, norm_value: torch.Tensor) -> torch.Tensor:
        """
        Undo the normalization done in the normalize() function.

        Parameters:
        -----------
        norm_value: Tensor [batch, series, time steps, samples]
            A tensor containing the normalized values to be denormalized.

        Returns:
        --------
        value: Tensor [batch, series, time steps, samples]
            The denormalized values.
        """
        return norm_value


class NormalizationStandardization:
    """
    Normalization helper for the standardization.

    The data for each batch and each series will be normalized by:
    - substracting the historical data mean,
    - and dividing by the historical data standard deviation.

    Use a lower bound of 1e-8 for the standard deviation to avoid numerical problems.
    """

    def __init__(self, hist_value: torch.Tensor, mean=None, std=None, mean_weight=None, std_weight=None):
        """
        Parameters:
        -----------
        hist_value: torch.Tensor [batch, series, time steps]
            Historical data which can be used in the normalization.
        """
        std, mean = torch.std_mean(hist_value, dim=2, unbiased=True, keepdim=True)
        self.std = std.clamp(min=1e-8)#torch.ones_like(std)#
        self.mean = mean

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        """
        Normalize the given values according to the historical data sent in the constructor.

        Parameters:
        -----------
        value: Tensor [batch, series, time steps]
            A tensor containing the values to be normalized.

        Returns:
        --------
        norm_value: Tensor [batch, series, time steps]
            The normalized values.
        """
        value = (value - self.mean) / self.std
        return value

    def denormalize(self, norm_value: torch.Tensor) -> torch.Tensor:
        """
        Undo the normalization done in the normalize() function.

        Parameters:
        -----------
        norm_value: Tensor [batch, series, time steps, samples]
            A tensor containing the normalized values to be denormalized.

        Returns:
        --------
        value: Tensor [batch, series, time steps, samples]
            The denormalized values.
        """
        norm_value = (norm_value * self.std[:, :, :, None]) + self.mean[:, :, :, None]
        return norm_value

class LearnedStandardization:
    """
    Normalization helper for the standardization.

    The data for each batch and each series will be normalized by:
    - substracting the historical data mean,
    - and dividing by the historical data standard deviation.

    Use a lower bound of 1e-8 for the standard deviation to avoid numerical problems.
    """

    def __init__(self, hist_value: torch.Tensor, mean:torch.Tensor, std: torch.Tensor, mean_weight: torch.Tensor, std_weight: torch.Tensor):
        """
        Parameters:
        -----------
        hist_value: torch.Tensor [batch, series, time steps]
            Historical data which can be used in the normalization.
        """
        hist_std, hist_mean = torch.std_mean(hist_value, dim=2, unbiased=True, keepdim=True)
        self.mean = (hist_mean + torch.exp(mean_weight) * mean)/(1.0 +  torch.exp(mean_weight))
        self.std = (hist_std.clamp(min=1e-8) + torch.exp(std_weight)*std.abs().clamp(min=1e-8))/(1.0 +  torch.exp(std_weight))

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        """
        Normalize the given values according to the historical data sent in the constructor.

        Parameters:
        -----------
        value: Tensor [batch, series, time steps]
            A tensor containing the values to be normalized.

        Returns:
        --------
        norm_value: Tensor [batch, series, time steps]
            The normalized values.
        """
        value = (value - self.mean) / self.std
        return value

    def denormalize(self, norm_value: torch.Tensor) -> torch.Tensor:
        """
        Undo the normalization done in the normalize() function.

        Parameters:
        -----------
        norm_value: Tensor [batch, series, time steps, samples]
            A tensor containing the normalized values to be denormalized.

        Returns:
        --------
        value: Tensor [batch, series, time steps, samples]
            The denormalized values.
        """
        norm_value = (norm_value * self.std[:, :, :, None]) + self.mean[:, :, :, None]
        return norm_value

class DeepAR(nn.Module):
    """
    The top-level module for TACTiS.

    The role of this module is to handle everything outside of the encoder and decoder.
    This consists mainly the data manipulation ahead of the encoder and after the decoder.
    """

    def __init__(
        self,
        num_series: int,
        hist_len: int,
        depth: int,
        mid_dim:int, 
        data_normalization: str = "none",
        loss_normalization: str = "series",
        grad_range:int = 1,
    ):
        """
        Parameters:
        -----------
        num_series: int
            Number of series of the data which will be sent to the model.
        series_embedding_dim: int
            The dimensionality of the per-series embedding.
        input_encoder_layers: int
            Number of layers in the MLP which encodes the input data.
        bagging_size: Optional[int], default to None
            If set, the loss() method will only consider a random subset of the series at each call.
            The number of series kept is the value of this parameter.
        input_encoding_normalization: bool, default to True
            If true, the encoded input values (prior to the positional encoding) are scaled
            by the square root of their dimensionality.
        data_normalization: str ["", "none", "standardization"], default to "series"
            How to normalize the input values before sending them to the model.
        loss_normalization: str ["", "none", "series", "timesteps", "both"], default to "series"
            Scale the loss function by the number of series, timesteps, or both.
        positional_encoding: Optional[Dict[str, Any]], default to None
            If set to a non-None value, uses a PositionalEncoding for the time encoding.
            The options sent to the PositionalEncoding is content of this dictionary.
        encoder: Optional[Dict[str, Any]], default to None
            If set to a non-None value, uses a Encoder as the encoder.
            The options sent to the Encoder is content of this dictionary.
        temporal_encoder: Optional[Dict[str, Any]], default to None
            If set to a non-None value, uses a TemporalEncoder as the encoder.
            The options sent to the TemporalEncoder is content of this dictionary.
        copula_decoder: Optional[Dict[str, Any]], default to None
            If set to a non-None value, uses a CopulaDecoder as the decoder.
            The options sent to the CopulaDecoder is content of this dictionary.
        gaussian_decoder: Optional[Dict[str, Any]], default to None
            If set to a non-None value, uses a GaussianDecoder as the decoder.
            The options sent to the GaussianDecoder is content of this dictionary.
        """
        super().__init__()

        data_normalization = data_normalization.lower()
        assert data_normalization in {"", "none", "standardization", "learned"}
        loss_normalization = loss_normalization.lower()
        assert loss_normalization in {"", "none", "series", "timesteps", "both"}

        self.num_series = num_series
        self.hist_len = hist_len
        self.depth = depth
        self.mid_dim=mid_dim
        self.loss_normalization = loss_normalization
        self.grad_range = min(grad_range, hist_len)

        self.learned_mean = torch.nn.Parameter(torch.zeros(1, num_series, 1))
        self.learned_std = torch.nn.Parameter(torch.ones(1, num_series, 1))
        self.mean_weight = torch.nn.Parameter(torch.zeros(1, num_series, 1))
        self.std_weight = torch.nn.Parameter(torch.zeros(1, num_series, 1))
        self.batch_num=0

        self.data_normalization = {
            "": NormalizationIdentity,
            "none": NormalizationIdentity,
            "standardization": NormalizationStandardization,
            "learned" : LearnedStandardization,
        }[data_normalization]



        elayers = nn.ModuleList([])
        for i in range(self.depth):
            if i == 0:
                elayers.append(
                    nn.Linear(self.num_series*self.hist_len, self.mid_dim)
                )  # +1 for the value, +1 for the mask, and the per series embedding
                elayers.append(nn.ReLU())
            else:
                elayers.append(nn.Linear(self.mid_dim, self.num_series))
        self.network = nn.Sequential(*elayers)

    def forward(self, hist_time: torch.Tensor, hist_value: torch.Tensor, pred_time: torch.Tensor, pred_value: torch.Tensor
    ) -> torch.Tensor:
        return self.loss(hist_time, hist_value, pred_time, pred_value)

    def loss(
        self, hist_time: torch.Tensor, hist_value: torch.Tensor, pred_time: torch.Tensor, pred_value: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the loss function of the model.

        Parameters:
        -----------
        hist_time: Tensor [batch, series, time steps] or [batch, 1, time steps] or [batch, time steps]
            A tensor containing the time steps associated with the values of hist_value.
            If the series dimension is singleton or missing, then the time steps are taken as constant across all series.
        hist_value: Tensor [batch, series, time steps]
            A tensor containing the values that will be available at inference time.
        pred_time: Tensor [batch, series, time steps] or [batch, 1, time steps] or [batch, time steps]
            A tensor containing the time steps associated with the values of pred_value.
            If the series dimension is singleton or missing, then the time steps are taken as constant across all series.
        pred_value: Tensor [batch, series, time steps]
            A tensor containing the values that the model should learn to forecast at inference time.

        Returns:
        --------
        loss: torch.Tensor []
            The loss function of TACTiS, with lower values being better. Averaged over batches.
        """
        num_batches = hist_value.shape[0]
        num_series = hist_value.shape[1]
        num_hist_timesteps = hist_value.shape[2]
        num_pred_timesteps = pred_value.shape[2]
        device = hist_value.device

        # The normalizer uses the same parameters for both historical and prediction values
        normalizer = self.data_normalization(hist_value, self.learned_mean, self.learned_std, self.mean_weight, self.std_weight)
        hist_value = normalizer.normalize(hist_value)
        pred_value = normalizer.normalize(pred_value)
        inputs = torch.cat([hist_value, pred_value], dim=-1).transpose(-1,-2)
        predicted_inputs = inputs.clone()
        grad_ind = torch.arange(self.grad_range, device = device).unsqueeze(0) + 1
        hist_ind = torch.arange(self.hist_len - self.grad_range, device =device).unsqueeze(0) +  + self.grad_range


        for i in range(num_pred_timesteps):
            pred_ind = num_hist_timesteps  + i
            pred_inputs = torch.cat([predicted_inputs[:, pred_ind - grad_ind], inputs[:, pred_ind - hist_ind]], dim=-2)
            predictions = self.network(pred_inputs.view(num_batches, num_series*self.hist_len))
            predicted_inputs[:, pred_ind, :] = predictions
        end_predictions = predicted_inputs.transpose(-1,-2)[:,:, num_hist_timesteps:]             

        diff = normalizer.denormalize(end_predictions.unsqueeze(-1)) - normalizer.denormalize(pred_value.unsqueeze(-1))
        loss = diff**2

        self.batch_num += 1

        


        if self.loss_normalization in {"series", "both"}:
            loss = loss / num_series
        if self.loss_normalization in {"timesteps", "both"}:
            loss = loss / num_pred_timesteps
        return loss.mean()

    def sample(
        self, num_samples: int, hist_time: torch.Tensor, hist_value: torch.Tensor, pred_time: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate the given number of samples from the forecasted distribution.

        Parameters:
        -----------
        num_samples: int
            How many samples to generate, must be >= 1.
        hist_time: Tensor [batch, series, time steps] or [batch, 1, time steps] or [batch, time steps]
            A tensor containing the times associated with the values of hist_value.
            If the series dimension is singleton or missing, then the time steps are taken as constant across all series.
        hist_value: Tensor [batch, series, time steps]
            A tensor containing the available values
        pred_time: Tensor [batch, series, time steps] or [batch, 1, time steps] or [batch, time steps]
            A tensor containing the times at which we want forecasts.
            If the series dimension is singleton or missing, then the time steps are taken as constant across all series.

        Returns:
        --------
        samples: torch.Tensor [batch, series, time steps, samples]
            Samples from the forecasted distribution.
        """
        num_batches = hist_value.shape[0]
        num_series = hist_value.shape[1]
        num_hist_timesteps = hist_value.shape[2]
        num_pred_timesteps = pred_time.shape[-1]
        device = hist_value.device

        # The normalizer uses the same parameters for both historical and prediction values
        normalizer = self.data_normalization(hist_value, self.learned_mean, self.learned_std, self.mean_weight, self.std_weight)
        orig_hist = hist_value
        hist_value = normalizer.normalize(hist_value)
        pred_value = torch.zeros(num_batches, num_series, num_pred_timesteps, device=device)
        inputs = torch.cat([hist_value, pred_value], dim=-1).transpose(-1,-2)
        hist_ind = torch.arange(self.hist_len, device =pred_value.device) + 1
        for i in range(num_pred_timesteps):
            pred_ind = num_hist_timesteps + i
            pred_inputs = inputs[:, pred_ind - hist_ind].view(num_batches,num_series*self.hist_len)
            inputs[:, pred_ind, :] = self.network(pred_inputs)
        inputs = inputs.transpose(-1, -2)
        pred_ind = num_hist_timesteps + torch.arange(num_pred_timesteps, device =device)

        samples = normalizer.denormalize(inputs.unsqueeze(-1)).repeat(1,1,1,num_samples)
        samples[:,:, pred_ind, :] += 1e-7*torch.randn([num_batches, num_series, num_pred_timesteps, num_samples], device=device)
        return samples