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

>> The various decoders for TACTiS, to output the forecasted distributions.
"""


import math
from typing import Any, Dict, Optional, Tuple, Type

import torch
from torch import nn
from torch.distributions import LowRankMultivariateNormal
from torch.nn import functional

from .marginal import DSFMarginal

def _compute_tree_depth(mask: torch.Tensor, times: torch.Tensor,cached=None) -> torch.Tensor:
    longest_gap = 144
    back_depth = 0
    time_window =48# 144
    if cached is None:

        time_ind = torch.argsort(times[0, :, :], dim=-1)
        time_rev = torch.argsort(time_ind, dim=-1)
        mask_depth = mask[0, :, :].clone()

        mask_depth = torch.gather(mask_depth, -1, time_ind)

        current_depth = 1
        max_depth = torch.zeros_like(mask_depth).int()
        prev_mask = torch.zeros_like(max_depth)
        next_mask = torch.zeros_like(max_depth)
        prev_values = torch.zeros_like(max_depth)
        next_values = torch.zeros_like(max_depth)

        mask_depth_copy = mask_depth.clone()

        mask_depth = torch.stack((mask_depth, torch.fliplr(mask_depth)), dim=0).int()
        mask_depth[:,:, 0] = 1 - mask_depth[:,:, 0]
        for i in range(mask_depth.shape[-1] -1):
            mask_depth[:,:,i+1] = (1 - mask_depth[:,:, i+1])*(1 +mask_depth[:,:, i] )
        mask_l = mask_depth[0]
        mask_r = torch.fliplr(mask_depth[1])

        current_indices =  torch.arange(times.shape[-1], device=times.device).unsqueeze(0).repeat(times.shape[1], 1).int()
        max_ind = current_indices.max()

        while (mask_l + mask_r - 1).max() > longest_gap:
            centers = mask_l == longest_gap
            left_ind = (mask_l >= longest_gap)
            right_ind = (mask_r + mask_l -1 >= longest_gap)*(mask_l <= longest_gap)
            next_mask[centers] = (current_indices + mask_r > max_ind)[centers].int()
            prev_values[centers] = current_indices[centers] - mask_l[centers]
            next_values[centers] = (current_indices[centers] + mask_r[centers])*(1-next_mask[centers])
            current_depth += 1
            current_val = mask_l+mask_r
            mask_l[left_ind] -= longest_gap
            mask_r[right_ind] -= (current_val - longest_gap)[right_ind]
            mask_r[centers] = 0
            mask_depth_copy[centers] = 1

        mask_depth = mask_depth_copy
        if (1-mask_depth[:, -1].int()).sum() > 0 or (1-mask_depth[:, 0].int()).sum() > 0:
            end_indices = mask_depth[:,-1] == 0
            start_indices = mask_depth[:,0] == 0
            prev_mask[:,0] = start_indices
            next_mask[:,-1] = end_indices

            first_hist = torch.arange(prev_mask.shape[1], 0, -1, device=prev_mask.device)
            last_hist = torch.arange(prev_mask.shape[1], device=prev_mask.device)
            first_hist = torch.argmax(first_hist*mask_depth, dim=-1)
            last_hist = torch.argmax(last_hist*mask_depth, dim=-1)
            next_values[start_indices,0] = first_hist[start_indices].int()
            prev_values[end_indices,-1] = last_hist[end_indices].int()


            mask_depth[:,0] = 1
            mask_depth[:,-1] = 1
            max_depth[end_indices.bool(), -1] +=current_depth
            max_depth[start_indices.bool(), 0] +=current_depth
            current_depth += 1

        mask_depth = torch.stack((mask_depth, torch.fliplr(mask_depth)), dim=0).int()
        mask_depth[:,:, 0] = 1 - mask_depth[:,:, 0]
        for i in range(mask_depth.shape[-1] -1):
            mask_depth[:,:,i+1] = (1 - mask_depth[:,:, i+1])*(1 +mask_depth[:,:, i] )
        mask_l = mask_depth[0]
        mask_r = torch.fliplr(mask_depth[1])
        current_indices =  torch.arange(times.shape[-1], device=times.device).unsqueeze(0).repeat(times.shape[1], 1).int()
        #print("next_mask", next_mask[0,:])
        #print("L", next_values[0,:])
        #print("R", prev_values[0,:])


        while mask_l.sum() > 0:
            half = (mask_l + mask_r)//2
            centers = (half == mask_l) * (half > 0)
            max_depth[centers] += current_depth
            prev_values[centers] = current_indices[centers] - mask_l[centers]
            next_values[centers] = current_indices[centers] + mask_r[centers]
            current_depth += 1
            odd_interval = torch.remainder(mask_l + mask_r + 1, 2)
            mask_l[mask_l >= half] -= half[mask_l >= half]
            mask_r[(mask_r > half - odd_interval)*( half > 0)] -= (half - odd_interval)[(mask_r > half - odd_interval)*( half > 0)] + 1 

        window_indices = current_indices - time_window
        window_mask = torch.logical_or(window_indices < 0, window_indices > max_ind)
        window_mask[:,time_window:] = torch.logical_or(window_mask[:,time_window:], max_depth[:,time_window:] > max_depth[:,0:-time_window])
        window_mask = window_mask.int() 
        window_indices = torch.clamp(window_indices, min=0, max=max_ind)

        if back_depth > 0:
            prev_near = torch.arange(-back_depth, 0, device=prev_values.device).repeat(prev_values.shape[0], prev_values.shape[1], 1)
            prev_near += prev_values.unsqueeze(-1)
            prev_near_mask = (prev_near < 0) + (prev_near > max_ind)
            prev_near = torch.clamp(prev_near, min=0, max=max_ind)
            prev_near = torch.gather(time_ind.unsqueeze(-1).repeat(1,1,back_depth), 1, prev_near.long())
            prev_near = torch.gather(prev_near, 1, time_rev.unsqueeze(-1).repeat(1,1,back_depth))

            next_near = torch.arange(1, back_depth+1, device=next_values.device).repeat(next_values.shape[0], next_values.shape[1], 1)
            next_near += next_values.unsqueeze(-1)
            next_near_mask = (next_near < 0) + (next_near > max_ind)
            next_near = torch.clamp(next_near, min=0, max=max_ind)
            next_near = torch.gather(time_ind.unsqueeze(-1).repeat(1,1,back_depth), 1, next_near.long())
            next_near = torch.gather(next_near, 1, time_rev.unsqueeze(-1).repeat(1,1,back_depth))

        max_depth = torch.gather(max_depth, -1, time_rev).float()
        next_mask = torch.gather(next_mask, -1, time_rev).float()
        prev_mask = torch.gather(prev_mask, -1, time_rev).float()
        window_mask = torch.gather(window_mask, -1, time_rev).float()
        prev_values = torch.gather(time_ind, -1, prev_values.long())
        prev_values = torch.gather(prev_values, -1, time_rev)
        next_values = torch.gather(time_ind, -1, next_values.long())
        next_values = torch.gather(next_values, -1, time_rev)
        window_indices = torch.gather(time_ind, -1, window_indices.long())
        window_indices = torch.gather(window_indices, -1, time_rev)
        offsets = prev_values.shape[1] * torch.arange(prev_values.shape[0], device=prev_values.device).unsqueeze(1)
        indices = torch.arange(prev_values.shape[1], device=prev_values.device).unsqueeze(0) + offsets
        prev_values += offsets
        next_values += offsets
        window_indices += offsets
        max_depth += ((time_ind)/2.0)/(2*float(times.shape[-1]))
        cached = {}
        cached['max_depth'] = max_depth.detach()
        cached['prev_values'] = prev_values.detach()
        cached['next_values'] = next_values.detach()
        cached['indices'] = indices.detach()
        cached['prev_mask'] = prev_mask.detach()
        cached['next_mask'] = next_mask.detach()
        cached['mask_depth'] = mask_depth.detach()
        cached['window_indices'] = window_indices.detach()
        cached['window_mask'] = window_mask.detach()
        if back_depth >0:
            cached['prev_near'] = prev_near.detach()
            cached['next_near'] = next_near.detach()
            cached['prev_near_mask'] = prev_near_mask.detach()
            cached['next_near_mask'] = next_near_mask.detach()

        
    
    else:
        max_depth = cached['max_depth']
        prev_values = cached['prev_values']
        next_values = cached['next_values']
        indices = cached['indices']
        prev_mask = cached['prev_mask']
        next_mask = cached['next_mask']
        mask_depth = cached['mask_depth']
        window_indices = cached['window_indices']
        window_mask = cached['window_mask']
        if back_depth > 0:
            prev_near = cached['prev_near']
            next_near = cached['next_near']
            prev_near_mask = cached['prev_near_mask']
            next_near_mask = cached['next_near_mask']
    max_depth += (torch.rand_like(max_depth)/2.0)/(2*float(times.shape[-1]))
    prev_values = prev_values.T.unsqueeze(0).repeat(prev_values.shape[0], 1,1)
    comp = max_depth.unsqueeze(2) > max_depth.T.unsqueeze(0) 
    next_values = comp*indices.T.unsqueeze(0) + (~comp)*next_values.T.unsqueeze(0)

    prev_role = prev_values.shape[0] * torch.arange(prev_values.shape[0],device=prev_values.device).unsqueeze(1) + torch.arange(prev_values.shape[0],device=prev_values.device).unsqueeze(0)
    next_role = prev_values.shape[0]**2 + prev_values.shape[0] * torch.arange(prev_values.shape[0],device=prev_values.device).unsqueeze(1) + torch.arange(prev_values.shape[0],device=prev_values.device).unsqueeze(0)
    prev_role = prev_role.unsqueeze(1).repeat(1, prev_values.shape[1],1).int()
    next_role = next_role.unsqueeze(1).repeat(1, prev_values.shape[1],1).int()
    window_role = 2*prev_values.shape[0]**2 + torch.arange(prev_values.shape[0],device=prev_values.device).int().unsqueeze(1).unsqueeze(2).repeat(1, prev_values.shape[1],1)
    

    prev_mask = prev_mask.T.unsqueeze(0).repeat(prev_mask.shape[0], 1,1)
    next_mask = comp*indices.T.unsqueeze(0) + (~comp)*next_mask.T.unsqueeze(0)
    

    if back_depth > 0:
        prev_diag = torch.diagonal(prev_values, dim1=0, dim2=-1).T
        prev_near_mask += mask_depth.flatten()[prev_diag].unsqueeze(-1) < mask_depth.flatten()[prev_near]
        prev_near_role = torch.diagonal(prev_role, dim1=0, dim2=-1).T.unsqueeze(-1).repeat(1,1,back_depth)
        prev_values = torch.cat([prev_values, prev_near], dim=-1)
        prev_mask = torch.cat([prev_mask, prev_near_mask], dim=-1)
        prev_role = torch.cat([prev_role, prev_near_role], dim=-1)

        next_diag = torch.diagonal(next_values, dim1=0, dim2=-1).T
        next_near_mask += mask_depth.flatten()[next_diag].unsqueeze(-1) < mask_depth.flatten()[next_near]
        next_near_role = torch.diagonal(next_role, dim1=0, dim2=-1).T.unsqueeze(-1).repeat(1,1,back_depth)
        next_values = torch.cat([next_values, next_near], dim=-1)
        next_mask = torch.cat([next_mask, next_near_mask], dim=-1)
        next_role = torch.cat([next_role, next_near_role], dim=-1)
    
    return max_depth, torch.cat((prev_values, next_values, window_indices.unsqueeze(-1)), dim=-1), torch.cat((prev_mask, next_mask, window_mask.unsqueeze(-1)), dim=-1), torch.cat((prev_role, next_role, window_role), dim=-1), cached


def _merge_series_time_dims(x: torch.Tensor) -> torch.Tensor:
    """
    Convert a Tensor with dimensions [batch, series, time steps, ...] to one with dimensions [batch, series * time steps, ...]
    """
    assert x.dim() >= 3
    return x.view((x.shape[0], x.shape[1] * x.shape[2]) + x.shape[3:])


def _split_series_time_dims(x: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """
    Convert a Tensor with dimensions [batch, series * time steps, ...] to one with dimensions [batch, series, time steps, ...]
    """
    assert x.dim() + 1 == len(target_shape)
    return x.view(target_shape)


def _easy_mlp(
    input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, activation: Type[nn.Module]
) -> nn.Sequential:
    """
    Generate a MLP with the given parameters.
    """
    elayers = [nn.Linear(input_dim, hidden_dim), activation()]
    for _ in range(1, num_layers):
        elayers += [nn.Linear(hidden_dim, hidden_dim), activation()]
    elayers += [nn.Linear(hidden_dim, output_dim)]
    return nn.Sequential(*elayers)


class CopulaDecoder(nn.Module):
    """
    A decoder which forecast using a distribution built from a copula and marginal distributions.
    """

    def __init__(
        self,
        input_dim: int,
        min_u: float = 0.0,
        max_u: float = 1.0,
        skip_sampling_marginal: bool = False,
        trivial_copula: Optional[Dict[str, Any]] = None,
        attentional_copula: Optional[Dict[str, Any]] = None,
        dsf_marginal: Optional[Dict[str, Any]] = None,
    ):
        """
        Parameters:
        -----------
        input_dim: int
            The dimension of the encoded representation (upstream data encoder).
        min_u: float, default to 0.0
        max_u: float, default to 1.0
            The values sampled from the copula will be scaled from [0, 1] to [min_u, max_u] before being sent to the marginal.
        skip_sampling_marginal: bool, default to False
            If set to True, then the output from the copula will not be transformed using the marginal during sampling.
            Does not impact the other transformations from observed values to the [0, 1] range.
        trivial_copula: Dict[str, Any], default to None
            If set to a non-None value, uses a TrivialCopula.
            The options sent to the TrivialCopula is content of this dictionary.
        attentional_copula: Dict[str, Any], default to None
            If set to a non-None value, uses a AttentionalCopula.
            The options sent to the AttentionalCopula is content of this dictionary.
        dsf_marginal: Dict[str, Any], default to None
            If set to a non-None value, uses a DSFMarginal.
            The options sent to the DSFMarginal is content of this dictionary.
        """
        super().__init__()

        assert (trivial_copula is not None) + (
            attentional_copula is not None
        ) == 1, "Must select exactly one type of copula"
        assert (dsf_marginal is not None) == 1, "Must select exactly one type of marginal"

        self.min_u = min_u
        self.max_u = max_u
        self.skip_sampling_marginal = skip_sampling_marginal

        if trivial_copula is not None:
            self.copula = TrivialCopula(**trivial_copula)
        if attentional_copula is not None:
            self.copula = AttentionalCopula(input_dim=input_dim, **attentional_copula)

        if dsf_marginal is not None:
            self.marginal = DSFMarginal(context_dim=input_dim, **dsf_marginal)
        self.cached = None

    def loss(self, encoded: torch.Tensor, mask: torch.BoolTensor, true_value: torch.Tensor, times: torch.Tensor,) -> torch.Tensor:
        """
        Compute the loss function of the decoder.

        Parameters:
        -----------
        encoded: Tensor [batch, series, time steps, embedding dimension]
            A tensor containing an embedding for each variable and time step.
            This embedding is coming from the encoder, so contains shared information across series and time steps.
        mask: BoolTensor [batch, series, time steps]
            A tensor containing a mask indicating whether a given value was available for the encoder.
            The decoder only forecasts values for which the mask is set to False.
        true_value: Tensor [batch, series, time steps]
            A tensor containing the true value for the values to be forecasted.
            Only the values where the mask is set to False will be considered in the loss function.

        Returns:
        --------
        loss: torch.Tensor [batch]
            The loss function, equal to the negative log likelihood of the distribution.

        """
        max_depth, att_hist, att_mask, att_role, cached = _compute_tree_depth(mask, times, cached=self.cached)
        self.cached = cached    
        

        max_depth = _merge_series_time_dims(max_depth.unsqueeze(0))
        att_hist = _merge_series_time_dims(att_hist.unsqueeze(0))
        att_mask = _merge_series_time_dims(att_mask.unsqueeze(0))
        att_role = _merge_series_time_dims(att_role.unsqueeze(0))
        times = _merge_series_time_dims(times[0].unsqueeze(0))

        encoded = _merge_series_time_dims(encoded)
        mask = _merge_series_time_dims(mask)
        true_value = _merge_series_time_dims(true_value)

        # Assume that the mask is constant inside the batch
        mask = mask[0, :]

        hist_encoded = encoded[:, mask, :]
        pred_encoded = encoded[:, ~mask, :]
        hist_true_x = true_value[:, mask]
        pred_true_x = true_value[:, ~mask]

        pred_depths = max_depth[0, :]
        pred_depths[mask] *= -1

        true_u = torch.zeros_like(true_value)

        # if encoded.device.index == 0:
        #     print(f"encoded: {encoded}\n true_value: {true_value}\n mask: {mask}\n times {times}")
        # Transform to [0,1] using the marginals
        hist_true_u = self.marginal.forward_no_logdet(hist_encoded, hist_true_x)
        pred_true_u, marginal_logdet = self.marginal.forward_logdet(pred_encoded, pred_true_x)
        true_u[:, mask] = hist_true_u
        true_u[:, ~mask] = pred_true_u

        num_variables = hist_encoded.shape[1]
        permutation = torch.argsort(pred_depths)[num_variables:]#torch.randperm(num_variables)#
        att_hist = att_hist[0, permutation, :]
        att_mask = att_mask[0, permutation, :].bool()
        att_role = att_role[0, permutation, :]
        copula_loss = self.copula.loss_local(
            encoded=encoded,
            true_u = true_u,
            permutation = permutation,
            history = att_hist,
            mask = att_mask,
            role = att_role,
            times=times,
        )
        # print(f"loss cl:{copula_loss} ml:{marginal_logdet}")
        # Loss = negative log likelihood
        return copula_loss - marginal_logdet

    def sample(
        self, num_samples: int, encoded: torch.Tensor, mask: torch.BoolTensor, true_value: torch.Tensor,  times: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate the given number of samples from the forecasted distribution.

        Parameters:
        -----------
        num_samples: int
            How many samples to generate, must be >= 1.
        encoded: Tensor [batch, series, time steps, embedding dimension]
            A tensor containing an embedding for each variable and time step.
            This embedding is coming from the encoder, so contains shared information across series and time steps.
        mask: BoolTensor [batch, series, time steps]
            A tensor containing a mask indicating whether a given value is masked (available) for the encoder.
            The decoder only forecasts values for which the mask is set to False.
        true_value: Tensor [batch, series, time steps]
            A tensor containing the true value for the values to be forecasted.
            The values where the mask is set to True will be copied as-is in the output.

        Returns:
        --------
        samples: torch.Tensor [batch, series, time steps, samples]
            Samples drawn from the forecasted distribution.
        """

        max_depth, att_hist, att_mask, att_role, cached = _compute_tree_depth(mask, times)             

        max_depth = _merge_series_time_dims(max_depth.unsqueeze(0))
        att_hist = _merge_series_time_dims(att_hist.unsqueeze(0))
        att_mask = _merge_series_time_dims(att_mask.unsqueeze(0))
        att_role = _merge_series_time_dims(att_role.unsqueeze(0))
        times = _merge_series_time_dims(times[0].unsqueeze(0))

        target_shape = torch.Size((true_value.shape[0], true_value.shape[1], true_value.shape[2], num_samples))

        encoded = _merge_series_time_dims(encoded)
        mask = _merge_series_time_dims(mask)
        true_value = _merge_series_time_dims(true_value)

        # Assume that the mask is constant inside the batch
        mask = mask[0, :]

        hist_encoded = encoded[:, mask, :]
        pred_encoded = encoded[:, ~mask, :]
        hist_true_x = true_value[:, mask]

        pred_depths = max_depth[0, :]
        pred_depths[mask] *= -1

        true_u = torch.zeros_like(true_value)

        # Transform to [0,1] using the marginals
        hist_true_u = self.marginal.forward_no_logdet(hist_encoded, hist_true_x)
        true_u[:, mask] = hist_true_u
        num_variables = hist_encoded.shape[1]
        permutation = torch.argsort(pred_depths)[num_variables:]
        att_hist = att_hist[0,:, :]
        att_role = att_role[0,:,:]
        att_mask = att_mask[0, :,:].bool()

        pred_samples = self.copula.sample_local(
            num_samples=num_samples,
            encoded=encoded,
            true_u=true_u,
            permutation = permutation,
            history = att_hist,
            mask = att_mask,
            role = att_role,
            times = times,
        )

        pred_samples=pred_samples[:, ~mask, :]
        if not self.skip_sampling_marginal:
            # Transform away from [0,1] using the marginals
            pred_samples = self.min_u + (self.max_u - self.min_u) * pred_samples
            pred_samples = self.marginal.inverse(
                pred_encoded,
                pred_samples,
            )

        samples = torch.zeros(
            target_shape[0], target_shape[1] * target_shape[2], target_shape[3], device=encoded.device
        )
        samples[:, mask, :] = hist_true_x[:, :, None]
        samples[:, ~mask, :] = pred_samples

        return _split_series_time_dims(samples, target_shape)



class AttentionalCopula(nn.Module):
    """
    A non-parametric copula based on attention between the various variables.
    """

    def __init__(
        self,
        input_dim: int,
        attention_heads: int,
        attention_layers: int,
        attention_dim: int,
        mlp_layers: int,
        mlp_dim: int,
        resolution: int = 10,
        dropout: float = 0.1,
        fixed_permutation: bool = False,
        num_series: int = 5,
    ):
        """
        Parameters:
        -----------
        input_dim: int
            Dimension of the encoded representation.
        attention_heads: int
            How many independant heads the attention layer will have. Each head will have its own independant MLP
            to generate the keys and values.
        attention_layers: int
            How many successive attention layers copula will use. Each layer will have its own independant MLPs
            to generate the keys and values.
        attention_dim: int
            The size of the attention layer output.
        mlp_layers: int
            The number of hidden layers in the MLP that produces the keys and values for the attention layer,
            and in the MLP that takes the attention output to generate the distribution parameter.
        mlp_dim: int
            The size of the hidden layers in the MLP that produces the keys and values for the attention layer,
            and in the MLP that takes the attention output to generate the distribution parameter.
        resolution: int, default to 10
            How many bins to pick from when sampling variables.
            Higher values are more precise, but slower to train.
        dropout: float, default to 0.1
            Dropout parameter for the attention.
        fixed_permutation: bool, default False
            If set to true, then the copula always use the same permutation, instead of using random ones.
        """
        super().__init__()

        self.input_dim = input_dim
        self.attention_heads = attention_heads
        self.attention_layers = attention_layers
        self.attention_dim = attention_dim
        self.mlp_layers = mlp_layers
        self.mlp_dim = mlp_dim
        self.resolution = resolution
        self.dropout = dropout
        self.fixed_permutation = fixed_permutation
        self.num_series = num_series

        # Parameters for the attention layers in the copula
        # For each layer and each head, we have two MLP to create the keys and values
        # After each layer, we transform the embedding using a feed-forward network, consisting of
        # two linear layer with a ReLu in-between both
        # At the very beginning, we have a linear layer to change the embedding to the proper dimensionality
        self.dimension_shifting_layer = nn.Linear(self.input_dim, self.attention_heads * self.attention_dim)
        self.role_bias =  nn.ModuleList([nn.Embedding(num_embeddings=self.num_series*(2*self.num_series + 1), embedding_dim=self.attention_heads) for _ in range(self.attention_layers)])
        self.time_net = nn.ModuleList(
            [
                _easy_mlp(
                    input_dim=1,
                    hidden_dim=self.mlp_dim,
                    output_dim=self.attention_heads,
                    num_layers=self.mlp_layers,
                    activation=nn.ReLU,
                )
                for _ in range(self.attention_layers)
            ]
        )
        # one per layer and per head
        # The key and value creators take the input embedding together with the sampled [0,1] value as an input
        self.key_creators = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        _easy_mlp(
                            input_dim=self.input_dim + 1,
                            hidden_dim=self.mlp_dim,
                            output_dim=self.attention_dim,
                            num_layers=self.mlp_layers,
                            activation=nn.ReLU,
                        )
                        for _ in range(self.attention_heads)
                    ]
                )
                for _ in range(self.attention_layers)
            ]
        )
        self.value_creators = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        _easy_mlp(
                            input_dim=self.input_dim + 1,
                            hidden_dim=self.mlp_dim,
                            output_dim=self.attention_dim,
                            num_layers=self.mlp_layers,
                            activation=nn.ReLU,
                        )
                        for _ in range(self.attention_heads)
                    ]
                )
                for _ in range(self.attention_layers)
            ]
        )

        # one per layer
        self.attention_dropouts = nn.ModuleList([nn.Dropout(self.dropout) for _ in range(self.attention_layers)])
        self.attention_layer_norms = nn.ModuleList(
            [nn.LayerNorm(self.attention_heads * self.attention_dim) for _ in range(self.attention_layers)]
        )
        self.feed_forwards = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.attention_heads * self.attention_dim, self.attention_heads * self.attention_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.attention_heads * self.attention_dim, self.attention_heads * self.attention_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(self.attention_layers)
            ]
        )
        self.feed_forward_layer_norms = nn.ModuleList(
            [nn.LayerNorm(self.attention_heads * self.attention_dim) for _ in range(self.attention_layers)]
        )

        # Parameter extractor for the categorical distribution
        self.dist_extractors = _easy_mlp(
            input_dim=self.attention_heads * self.attention_dim,
            hidden_dim=self.mlp_dim,
            output_dim=self.resolution,
            num_layers=self.mlp_layers,
            activation=nn.ReLU,
        )

    def loss_local(
        self,
        encoded: torch.Tensor,
        true_u: torch.Tensor,
        permutation: torch.Tensor,
        history: torch.Tensor,
        mask: torch.Tensor, 
        role: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the loss function of the copula portion of the decoder.

        Parameters:
        -----------
        hist_encoded: Tensor [batch, series * time steps, embedding dimension]
            A tensor containing an embedding for each series and time step that does not have to be forecasted.
            The series and time steps dimensions are merged.
        hist_true_u: Tensor [batch, series * time steps]
            A tensor containing the true value for the values that do not have to be forecasted, transformed by the marginal distribution into U(0,1) values.
            The series and time steps dimensions are merged.
        pred_encoded: Tensor [batch, series * time steps, embedding dimension]
            A tensor containing an embedding for each variable and time step that does have to be forecasted.
            The series and time steps dimensions are merged.
        pred_true_u: Tensor [batch, series * time steps]
            A tensor containing the true value for the values to be forecasted, transformed by the marginal distribution into U(0,1) values.
            The series and time steps dimensions are merged.

        Returns:
        --------
        loss: torch.Tensor [batch]
            The loss function, equal to the negative log likelihood of the copula.
        """
        device = encoded.device

        # Permute the variables according the random permutation
        pred_encoded = encoded[:, permutation, :]
        pred_true_u = true_u[:, permutation]
        times = times[0].to(torch.get_default_dtype())
        pred_times = times[permutation]
        

        # The MLP which generates the keys and values used the encoded embedding + transformed true values.
        key_value_input = torch.cat([encoded, true_u[:, :, None]], axis=2)

        keys = [
            torch.cat([mlp(key_value_input)[:, None, :, :] for mlp in self.key_creators[layer]], axis=1)
            for layer in range(self.attention_layers)
        ]
        values = [
            torch.cat([mlp(key_value_input)[:, None, :, :] for mlp in self.value_creators[layer]], axis=1)
            for layer in range(self.attention_layers)
        ]

        # During attention, we will add -float("inf") to pairs of indices where the variable to be forecasted (query)
        # is after the variable that gives information (key), after the random permutation.
        # Doing this prevent information from flowing from later in the permutation to before in the permutation,
        # which cannot happen during inference.
        # tril fill the diagonal and values that are below it, flip rotates it by 180 degrees,
        # leaving only the pairs of indices which represent not yet sampled values.
        # Note float("inf") * 0 is unsafe, so do the multiplication inside the torch.tril()
        # pred/hist_encoded dimensions: number of batches, number of variables, size of embedding per variable
        '''
        product_mask = torch.ones(
            num_batches,
            self.attention_heads,
            num_variables,
            num_variables + num_history,
            device=device,
        )
        product_mask = torch.tril(float("inf") * product_mask).flip((2, 3))
        '''

        # At the beginning of the attention, we start with the input embedding.
        # Since it does not necessarily have the same dimensions as the hidden layers, apply a linear layer to scale it up.
        att_value = self.dimension_shifting_layer(pred_encoded)

        for layer in range(self.attention_layers):
            # Split the hidden layer into its various heads
            att_value_heads = att_value.reshape(
                att_value.shape[0], att_value.shape[1], self.attention_heads, self.attention_dim
            )
            #print(torch.einsum("bvhi,bhvwi->bhvw", att_value_heads, keys[layer][:,:,history]).shape)

            # Attention layer, for each batch and head:
            # A_vi' = sum_w(softmax_w(sum_i(Q_vi * K_wi) / sqrt(d)) * V_wi')

            # Einstein sum indices:
            # b: batch number
            # h: attention head number (Note the change in order for att_value_heads)
            # v: variable we want to predict
            # w: variable we want to get information from (history or prediction)
            # i: embedding dimension of the keys and queries (self.attention_dim)
            product_base = torch.einsum("bvhi,bhvwi->bhvw", att_value_heads, keys[layer][:,:,history])
            product_bias = torch.einsum("bvwh->bhvw", self.role_bias[layer](role).unsqueeze(0))
            # print("last:", times[history].dtype, pred_times.dtype)
            time_bias = torch.einsum("bvwh->bhvw", self.time_net[layer]((times[history] - pred_times.unsqueeze(1)).unsqueeze(-1)).unsqueeze(0))

            # Adding -inf shunts the attention to zero, for any variable that has not "yet" been predicted,
            # aka: are in the future according to the permutation.
            product = product_base + product_bias + time_bias #- product_mask
            product[:,:, mask] -= float("inf")
            product = self.attention_dim ** (-0.5) * product
            weights = nn.functional.softmax(product, dim=-1)

            # Einstein sum indices:
            # b: batch number
            # h: attention head number (Note the change in order for the result)
            # v: variable we want to predict
            # w: variable we want to get information from (history or prediction)
            # j: embedding dimension of the values (self.attention_dim)
            att = torch.einsum("bhvw,bhvwj->bvhj", weights, values[layer][:,:,history])

            # Merge back the various heads to allow the feed forwards module to share information between heads
            att_merged_heads = att.reshape(att.shape[0], att.shape[1], att.shape[2] * att.shape[3])
            att_merged_heads = self.attention_dropouts[layer](att_merged_heads)
            att_value = att_value + att_merged_heads
            att_value = self.attention_layer_norms[layer](att_value)
            att_feed_forward = self.feed_forwards[layer](att_value)
            att_value = att_value + att_feed_forward
            att_value = self.feed_forward_layer_norms[layer](att_value)

        # Compute the logarithm likelihood of the conditional distribution.
        # Note: This section could instead call a specialized module to allow for easier customization.
        # Get conditional distributions over bins for all variables but the first one.
        # The first one is considered to always be U(0,1), which has a constant logarithm likelihood of 0.
        logits = self.dist_extractors(att_value)[:, :, :]#self.dist_extractors(att_value)[:, 1:, :]

        # Assign each observed U(0,1) value to a bin. The clip is to avoid issues with numerical inaccuracies.
        #target = torch.clip(torch.floor(pred_true_u[:, 1:] * self.resolution).long(), min=0, max=self.resolution - 1)
        target = torch.clip(torch.floor(pred_true_u[:, :] * self.resolution).long(), min=0, max=self.resolution - 1)

        # We multiply the probability by self.resolution to get the PDF of the continuous-by-part distribution.
        logprob = math.log(self.resolution) + nn.functional.log_softmax(logits, dim=2)
        # For each batch + variable pair, we want the value of the logits associated with its true value (target):
        # logprob[batch,variable] = logits[batch,variable,target[batch,variable]]
        # Since gather wants the same number of dimensions for both tensors, add and remove a dummy third dimension.
        logprob = torch.gather(logprob, dim=2, index=target[:, :, None])[:, :, 0]

        return -logprob.sum(axis=1)  # Only keep the batch dimension


    def sample_local(
        self, num_samples: int, encoded: torch.Tensor, true_u: torch.Tensor,  permutation : torch.Tensor, history : torch.Tensor, mask:torch.Tensor, role:torch.Tensor, times:torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate the given number of samples from the forecasted copula.

        Parameters:
        -----------
        num_samples: int
            How many samples to generate, must be >= 1.
        hist_encoded: Tensor [batch, series * time steps, embedding dimension]
            A tensor containing an embedding for each variable and time step that does not have to be forecasted.
            The series and time steps dimensions are merged.
        hist_true_u: Tensor [batch, series * time steps]
            A tensor containing the true value for the values that do not have to be forecasted, transformed by the marginal distribution into U(0,1) values.
            The series and time steps dimensions are merged.
        pred_encoded: Tensor [batch, series * time steps, embedding dimension]
            A tensor containing an embedding for each variable and time step that does have to be forecasted.
            The series and time steps dimensions are merged.

        Returns:
        --------
        samples: torch.Tensor [batch, series * time steps, samples]
            Samples drawn from the forecasted copula, thus in the [0, 1] range.
            The series and time steps dimensions are merged.
        """
        num_batches = encoded.shape[0]
        num_variables = encoded.shape[1]
        num_gen = permutation.shape[-1]
        device = encoded.device
        times = times[0].float()
        pred_times = times[permutation]

        # The MLP which generates the keys and values used the encoded embedding + transformed true values.
        key_value_input = torch.cat([encoded, true_u[:, :, None]], axis=2)
        keys = [
            torch.cat([mlp(key_value_input)[:, None, :, :] for mlp in self.key_creators[layer]], axis=1)
            for layer in range(self.attention_layers)
        ]
        values = [
            torch.cat([mlp(key_value_input)[:, None, :, :] for mlp in self.value_creators[layer]], axis=1)
            for layer in range(self.attention_layers)
        ]

        # We will store the keys and values from the sampled variables as we do the sampling
        samples = torch.zeros(num_batches, num_variables, num_samples).to(device)
        for i in range(len(keys)):
            keys[i][:,:, permutation] = torch.zeros_like(keys[i][:,:, permutation])
            keys[i] = keys[i].unsqueeze(1).repeat(1,num_samples,1,1,1)
        for i in range(len(values)):
            values[i][:,:, permutation] = torch.zeros_like(values[i][:,:, permutation])
            values[i] = values[i].unsqueeze(1).repeat(1,num_samples,1,1,1)

        # We sample the copula one variable at a time, following the order from the drawn permutation.
        for i in range(num_gen):
            # Vector containing which variable we sample at this step of the copula.
            p = permutation[i]
            # Note that second dimension here no longer represent the variables (as in the loss method), but the samples.
            current_pred_encoded = encoded[:,p, :].unsqueeze(1).repeat(1,num_samples, 1)

            if False:#i == 0:
                # By construction, the first variable to be sampled is always sampled according to a Uniform(0,1).
                current_samples = torch.rand(num_batches, num_samples, device=device)
            else:
                att_value = self.dimension_shifting_layer(current_pred_encoded)

                for layer in range(self.attention_layers):
                    # Split the hidden layer into its various heads
                    att_value_heads = att_value.reshape(
                        att_value.shape[0], att_value.shape[1], self.attention_heads, self.attention_dim
                    )

                    # Calculate attention weights
                    # Einstein sum indices:
                    # b: batch number
                    # n: sample number
                    # h: attention head number
                    # w: variable we want to get information from (history or prediction)
                    # i: embedding dimension of the keys and queries (self.input_dim)
                    product = torch.einsum("bnhi,bnhwi->bnhw", att_value_heads, keys[layer][:,:,:,history[p],:])
                    product_bias = torch.einsum("bnwh->bnhw", self.role_bias[layer](role[p]).unsqueeze(0).unsqueeze(1))
                    time_bias = torch.einsum("bnwh->bnhw", self.time_net[layer]((times[history[p]] - times[p]).unsqueeze(-1)).unsqueeze(0).unsqueeze(1))
                    product += product_bias + time_bias
                    product[:,:,:, mask[p]] -= float("inf")
                    product = self.attention_dim ** (-0.5) * product

                    weights = nn.functional.softmax(product, dim=3)

                    # Get attention representation using weights (for conditional distribution)
                    # Einstein sum indices:
                    # b: batch number
                    # n: sample number
                    # h: attention head number
                    # w: variable we want to get information from (history or prediction)
                    # j: embedding dimension of the values (self.hid_dim)
                    att = torch.einsum("bnhw,bnhwj->bnhj", weights, values[layer][:,:,:,history[p],:])

                    # Merge back the various heads to allow the feed forwards module to share information between heads
                    att_merged_heads = att.reshape(att.shape[0], att.shape[1], att.shape[2] * att.shape[3])
                    att_merged_heads = self.attention_dropouts[layer](att_merged_heads)
                    att_value = att_value + att_merged_heads
                    att_value = self.attention_layer_norms[layer](att_value)
                    att_feed_forward = self.feed_forwards[layer](att_value)
                    att_value = att_value + att_feed_forward
                    att_value = self.feed_forward_layer_norms[layer](att_value)

                # Get the output distribution parameters
                logits = self.dist_extractors(att_value).reshape(num_batches * num_samples, self.resolution)
                # Select a single variable in {0, 1, 2, ..., self.resolution-1} according to the probabilities from the softmax
                current_samples = torch.multinomial(input=torch.softmax(logits, dim=1), num_samples=1)
                # Each point in the same bucket is equiprobable, and we used a floor function in the training
                current_samples = current_samples + torch.rand(*current_samples.shape).to(device)
                # Normalize to a variable in the [0, 1) range
                current_samples /= self.resolution
                current_samples = current_samples.reshape(num_batches, num_samples)

            # Compute the key and value associated with the newly sampled variable, for the attention of the next ones.
            key_value_input = torch.cat([current_pred_encoded, current_samples[:, :, None]], axis=-1)
            for layer in range(self.attention_layers):
                new_keys = torch.cat([k(key_value_input)[:, :, None, :] for k in self.key_creators[layer]], axis=2)
                new_values = torch.cat([v(key_value_input)[:, :, None, :] for v in self.value_creators[layer]], axis=2)
                keys[layer][:, :, :, p, :] = new_keys
                values[layer][:, :, :, p, :] = new_values

            # Collate the results, reversing the effect of the permutation
            # By using two lists of equal lengths, the resulting slice will be 2d, not 3d.
            samples[:, p, range(num_samples)] = current_samples

        return samples


class TrivialCopula(nn.Module):
    """
    The trivial copula where all variables are independent.
    """

    def __init__(self):
        super().__init__()

    def loss(
        self,
        hist_encoded: torch.Tensor,
        hist_true_u: torch.Tensor,
        pred_encoded: torch.Tensor,
        pred_true_u: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the loss function of the copula portion of the decoder.

        Parameters:
        -----------
        hist_encoded: Tensor [batch, series * time steps, embedding dimension]
            A tensor containing an embedding for each variable and time step that does not have to be forecasted.
            The series and time steps dimensions are merged.
        hist_true_u: Tensor [batch, series * time steps]
            A tensor containing the true value for the values that do not have to be forecasted, transformed by the marginal distribution into U(0,1) values.
            The series and time steps dimensions are merged.
        pred_encoded: Tensor [batch, series * time steps, embedding dimension]
            A tensor containing an embedding for each variable and time step that does have to be forecasted.
            The series and time steps dimensions are merged.
        pred_true_u: Tensor [batch, series * time steps]
            A tensor containing the true value for the values to be forecasted, transformed by the marginal distribution into U(0,1) values.
            The series and time steps dimensions are merged.

        Returns:
        --------
        embedding: torch.Tensor [batch]
            The loss function, equal to the negative log likelihood of the copula.
            This is always equal to zero.
        """
        batch_size = hist_encoded.shape[0]
        device = hist_encoded.device
        # Trivially, the probability of all u is equal to 1 if in the unit cube (which it should always be by construction)
        return torch.zeros(batch_size, device=device)

    def sample(
        self, num_samples: int, hist_encoded: torch.Tensor, hist_true_u: torch.Tensor, pred_encoded: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate the given number of samples from the trivial copula.

        Parameters:
        -----------
        num_samples: int
            How many samples to generate, must be >= 1.
        hist_encoded: Tensor [batch, series * time steps, embedding dimension]
            A tensor containing an embedding for each variable and time step that does not have to be forecasted.
            The series and time steps dimensions are merged.
        hist_true_u: Tensor [batch, series * time steps]
            A tensor containing the true value for the values that do not have to be forecasted, transformed by the marginal distribution into U(0,1) values.
            The series and time steps dimensions are merged.
        pred_encoded: Tensor [batch, series * time steps, embedding dimension]
            A tensor containing an embedding for each variable and time step that does have to be forecasted.
            The series and time steps dimensions are merged.

        Returns:
        --------
        samples: torch.Tensor [batch, series * time steps, samples]
            Samples drawn from the trivial copula, which is equal to the multi-dimensional Uniform(0, 1) distribution.
            The series and time steps dimensions are merged.
        """
        num_batches, num_variables, _ = pred_encoded.shape
        device = pred_encoded.device
        return torch.rand(num_batches, num_variables, num_samples, device=device)


class GaussianDecoder(nn.Module):
    """
    A decoder which forecast using a low-rank multivariate Gaussian distribution.
    """

    def __init__(
        self,
        input_dim: int,
        matrix_rank: int,
        mlp_layers: int,
        mlp_dim: int,
        min_d: float = 0.01,
        max_v: float = 50.0,
    ):
        """
        Parameters:
        -----------
        input_dim: int
            Dimension of the encoded representation.
        matrix_rank: int
            Rank of the covariance matrix, prior to adding its diagonal component.
        mlp_layers: int
            The number of hidden layers in the MLP that produces the components of the covariance matrix.
        mlp_dim: int
            The size of the hidden layers in the MLP that produces the components of the covariance matrix.
        min_d: float, default to 0.01
            Minimum value of the diagonal component of the covariance matrix.
            Too low values can lead to exceptions due to numerical errors.
        max_v: float, default to 50.0
            Maximum weight of the contribution from the latent variables to the observed variables.
            Too high values can lead to exceptions due to numerical errors.
        """
        super().__init__()

        self.input_dim = input_dim
        self.matrix_rank = matrix_rank
        self.mlp_layers = mlp_layers
        self.mlp_dim = mlp_dim
        self.min_d = min_d
        self.max_v = max_v

        # The covariance matrix low-rank approximation is as such:
        # Cov = V * V^t + d
        # Where V is a number of variables * matrix rank rectangular matrix, and d is diagonal.
        # This gives the same covariance as having matrix rank latent Normal(0,1) variables, and generating the output as:
        # output_i = sum_j V_ij latent_j + N(0, d_i)
        self.param_V_extractor = _easy_mlp(
            input_dim=self.input_dim,
            hidden_dim=self.mlp_dim,
            output_dim=self.matrix_rank,
            num_layers=self.mlp_layers,
            activation=nn.ReLU,
        )
        self.param_d_extractor = _easy_mlp(
            input_dim=self.input_dim,
            hidden_dim=self.mlp_dim,
            output_dim=1,
            num_layers=self.mlp_layers,
            activation=nn.ReLU,
        )
        self.param_mean_extractor = _easy_mlp(
            input_dim=self.input_dim,
            hidden_dim=self.mlp_dim,
            output_dim=1,
            num_layers=self.mlp_layers,
            activation=nn.ReLU,
        )

    def extract_params(self, pred_encoded: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the parameters of the low-rank Gaussian distribution.

        Parameters:
        -----------
        pred_encoded: Tensor [batch, series * time steps, embedding dimension]
            A tensor containing an embedding for each variable and time step.

        Returns:
        --------
        param_mean: torch.Tensor [batch, series * time steps]
            The mean of the Gaussian distribution
        param_d: torch.Tensor [batch, series * time steps]
            The diagonal of the Gaussian distribution covariance matrix
        param_V: torch.Tensor [batch, series * time steps, matrix rank]
            The contribution to each variable from each latent variable
        """
        # The last dimension of the mean and d parameters is a dummy one
        param_mean = self.param_mean_extractor(pred_encoded)[:, :, 0]
        # Parametrized covariance matrix d + V*V^t
        # This is the same parametrization as what Salinas et al. (2019) used for the covariance matrix.
        # An upper bound of the condition number of the matrix that will be used in the logdet or Cholesky is:
        # 1 + hid_dim * max_v^2 / min_d
        # We add these limits since a condition number near or above 2^23 will lead to grave numerical instability in the Cholesky decomposition.
        param_d = functional.softplus(self.param_d_extractor(pred_encoded))[:, :, 0]
        param_d = param_d + self.min_d
        param_V = self.param_V_extractor(pred_encoded)
        param_V = torch.tanh(param_V / self.max_v) * self.max_v

        return param_mean, param_d, param_V

    def loss(self, encoded: torch.Tensor, mask: torch.BoolTensor, true_value: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss function of the decoder.

        Parameters:
        -----------
        encoded: Tensor [batch, series, time steps, embedding dimension]
            A tensor containing an embedding for each variable and time step.
            This embedding is coming from the encoder, so contains shared information across series and time steps.
        mask: BoolTensor [batch, series, time steps]
            A tensor containing a mask indicating whether a given value was available for the encoder.
            The decoder only forecasts values for which the mask is set to False.
        true_value: Tensor [batch, series, time steps]
            A tensor containing the true value for the values to be forecasted.
            Only the values where the mask is set to False will be considered in the loss function.

        Returns:
        --------
        embedding: torch.Tensor [batch]
            The loss function, equal to the negative log likelihood of the distribution.
        """
        encoded = _merge_series_time_dims(encoded)
        mask = _merge_series_time_dims(mask)
        true_value = _merge_series_time_dims(true_value)

        # Assume that the mask is constant inside the batch
        mask = mask[0, :]

        # Ignore the encoding from the historical variables, since there are no interaction between the variables in this decoder.
        pred_encoded = encoded[:, ~mask, :]
        pred_true_x = true_value[:, ~mask]

        param_mean, param_d, param_V = self.extract_params(pred_encoded)
        log_prob = LowRankMultivariateNormal(loc=param_mean, cov_factor=param_V, cov_diag=param_d).log_prob(pred_true_x)

        return -log_prob

    def sample(
        self, num_samples: int, encoded: torch.Tensor, mask: torch.BoolTensor, true_value: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate the given number of samples from the forecasted distribution.

        Parameters:
        -----------
        num_samples: int
            How many samples to generate, must be >= 1.
        encoded: Tensor [batch, series, time steps, embedding dimension]
            A tensor containing an embedding for each variable and time step.
            This embedding is coming from the encoder, so contains shared information across series and time steps.
        mask: BoolTensor [batch, series, time steps]
            A tensor containing a mask indicating whether a given value was available for the encoder.
            The decoder only forecasts values for which the mask is set to False.
        true_value: Tensor [batch, series, time steps]
            A tensor containing the true value for the values to be forecasted.
            The values where the mask is set to True will be copied as-is in the output.

        Returns:
        --------
        samples: torch.Tensor [batch, series, time steps, samples]
            Samples drawn from the forecasted distribution.
        """
        num_batches = encoded.shape[0]
        num_series = encoded.shape[1]
        num_timesteps = encoded.shape[2]
        device = encoded.device

        encoded = _merge_series_time_dims(encoded)
        mask = _merge_series_time_dims(mask)
        true_value = _merge_series_time_dims(true_value)

        # Assume that the mask is constant inside the batch
        mask = mask[0, :]

        # Ignore the encoding from the historical variables, since there are no interaction between the variables in this decoder.
        pred_encoded = encoded[:, ~mask, :]
        # Except what is needed to copy to the output
        hist_true_x = true_value[:, mask]

        param_mean, param_d, param_V = self.extract_params(pred_encoded)

        dist = LowRankMultivariateNormal(loc=param_mean, cov_factor=param_V, cov_diag=param_d)
        # rsamples have the samples as the first dimension, so send it to the last dimension
        pred_samples = dist.rsample((num_samples,)).permute((1, 2, 0))

        samples = torch.zeros(num_batches, num_series * num_timesteps, num_samples, device=device)
        samples[:, mask, :] = hist_true_x[:, :, None]
        samples[:, ~mask, :] = pred_samples

        return _split_series_time_dims(samples, torch.Size((num_batches, num_series, num_timesteps, num_samples)))
