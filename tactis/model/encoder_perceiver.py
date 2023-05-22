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

>> The two possible encoders for TACTiS, based on the Transformer architecture.
"""
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from perceiver.model.core import PerceiverEncoder, InputAdapter

class IdentityAdapter(InputAdapter):
    def __init__(self, num_input_channels: int):
        """Transforms and position-encodes task-specific input to generic encoder input.
        :param num_input_channels: Number of channels of the generic encoder input produced by this adapter.
        """
        super().__init__(num_input_channels)

    def forward(self, x):
        return x

class Encoder(nn.Module):
    """
    The traditional encoder for TACTiS, based on the Transformer architecture.

    The encoder receives an input which contains for each series and time step:
    * The series value at the time step, masked to zero if part of the values to be forecasted
    * The mask
    * The embedding for the series
    * The embedding for the time step
    And has already been through any input encoder.

    The decoder returns an output containing an embedding for each series and time step.
    """

    def __init__(
        self,
        embedding_dim: int,
        perceiver_encoder: Optional[Dict[str, Any]] = None,
    ):
        """
        Parameters:
        -----------
        attention_layers: int
            How many successive attention layers this encoder will use.
        attention_heads: int
            How many independant heads the attention layer will have.
        attention_dim: int
            The size of the attention layer input and output, for each head.
        attention_feedforward_dim: int
            The dimension of the hidden layer in the feed forward step.
        dropout: float, default to 0.1
            Dropout parameter for the attention.
        """
        super().__init__()

        self.embedding_dim = embedding_dim

        self.transformer_encoder = PerceiverEncoder(input_adapter=IdentityAdapter(self.embedding_dim), **perceiver_encoder)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        Compute the embedding for each series and time step.

        Parameters:
        -----------
        encoded: Tensor [batch, series, time steps, input embedding dimension]
            A tensor containing an embedding for each series and time step.
            This embedding is expected to only contain local information, with no interaction between series or time steps.

        Returns:
        --------
        output: torch.Tensor [batch, series, time steps, output embedding dimension]
            The transformed embedding for each series and time step.
        """
        num_batches = encoded.shape[0]
        num_series = encoded.shape[1]
        num_timesteps = encoded.shape[2]

        # Merge the series and time steps, since the PyTorch attention implementation only accept three-dimensional input,
        # and the attention is applied between all tokens, no matter their series or time step.
        encoded = encoded.view(num_batches, num_series * num_timesteps, self.embedding_dim)

        output = self.transformer_encoder(encoded)

        return output
