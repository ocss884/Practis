import numpy as np
import torch
import torch.nn as nn
from gluonts.dataset.field_names import FieldName
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.util import copy_parameters
from gluonts.transform import (
    AddObservedValuesIndicator,
    CDFtoGaussianTransform,
    Chain,
    InstanceSampler,
    InstanceSplitter,
    RenameFields,
    TestSplitSampler,
    cdf_to_gaussian_forward_transform,
)
from pts.model.utils import get_module_forward_input_names
transformation = Chain(
            [
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),
            ]
        )

class SingleInstanceSampler(InstanceSampler):
    """
    Randomly pick a single valid window in the given time series.
    This fix the bias in ExpectedNumInstanceSampler which leads to varying sampling frequency
    of time series of unequal length, not only based on their length, but when they were sampled.
    """

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        a, b = self._get_bounds(ts)
        window_size = b - a + 1

        if window_size <= 0:
            return np.array([], dtype=int)

        indices = np.random.randint(window_size, size=1)
        return indices + a

def create_instance_splitter(mode: str, num_series, history_length, prediction_length, cdf_normalization: bool = False) :
        assert mode in ["training", "validation", "test"]

        if mode == "training":
            instance_sampler = SingleInstanceSampler(
                min_past=history_length,  # Will not pick incomplete sequences
                min_future=prediction_length,
            )
        elif mode == "validation":
            instance_sampler = SingleInstanceSampler(
                min_past=history_length,  # Will not pick incomplete sequences
                min_future=prediction_length,
            )
        elif mode == "test":
            # This splitter takes the last valid window from each multivariate series,
            # so any multi-window split must be done in the data definition.
            instance_sampler = TestSplitSampler()

        if cdf_normalization:
            normalize_transform = CDFtoGaussianTransform(
                cdf_suffix="_norm",
                target_field=FieldName.TARGET,
                target_dim=num_series,
                max_context_length=history_length,
                observed_values_field=FieldName.OBSERVED_VALUES,
            )
        else:
            normalize_transform = RenameFields(
                {
                    f"past_{FieldName.TARGET}": f"past_{FieldName.TARGET}_norm",
                    f"future_{FieldName.TARGET}": f"future_{FieldName.TARGET}_norm",
                }
            )

        return (
            InstanceSplitter(
                target_field=FieldName.TARGET,
                is_pad_field=FieldName.IS_PAD,
                start_field=FieldName.START,
                forecast_start_field=FieldName.FORECAST_START,
                instance_sampler=instance_sampler,
                past_length=history_length,
                future_length=prediction_length,
                time_series_fields=[FieldName.OBSERVED_VALUES],
            )
            + normalize_transform
        )

def create_predictor(train_network, prediction_network, num_series, history_length, prediction_length, cdf_normalization: bool = False):

    copy_parameters(train_network, prediction_network)

    output_transform = cdf_to_gaussian_forward_transform if cdf_normalization else None
    input_names = get_module_forward_input_names(prediction_network)
    prediction_splitter = create_instance_splitter("test", num_series, history_length, prediction_length, cdf_normalization)

    predictor = PyTorchPredictor(
        input_transform = transformation + prediction_splitter,
        output_transform = output_transform,
        input_names = input_names,
        prediction_net = prediction_network,
        batch_size = 48,
        prediction_length = prediction_length,
        device = "cuda"
        )
    return predictor