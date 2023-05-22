# Configuration
import os
import sys
REPO_NAME = "Practis"
def get_repo_basepath():
    cd = os.path.abspath(os.curdir)
    return cd[:cd.index(REPO_NAME) + len(REPO_NAME)]
REPO_BASE_PATH = get_repo_basepath()
sys.path.append(REPO_BASE_PATH)

import torch
import numpy as np
from tactis.gluon.network_perceiver import TACTiSPTrainingNetwork, TACTiSPPredictionNetwork
from tactis.gluon.dataset import generate_backtesting_datasets

import os
import json
import deepspeed
import argparse
from tqdm.auto import tqdm

def add_argument():
    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')

    parser.add_argument('--log-epoch-interval',
                        type=int,
                        default=5,
                        help="output logging information at a given interval")
    
    parser.add_argument("-f",
                        "--file", 
                        dest="file_path", 
                        required=True,
                        help="Path to the JSON file containing the parameters")

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    return args

######################## Load Parameters ########################

args = add_argument()

if os.path.exists(args.file_path):
# Read the JSON file and load its contents into a Python dictionary
    with open(args.file_path, "r") as json_file:
        params = json.load(json_file)
        print("Parameters loaded from the JSON file:")
        # print(params)
else:
    print(f"File not found: {args.file_path}")

deepspeed.init_distributed()

######################## Load Data ########################

history_length = 2
prediction_length = 2
total_length = history_length+prediction_length 
backtest_id = 2
assert args.dataset_name in ["electricity_hourly", "fred_md", "traffic"], f"Invalid dataset name: {params['dataset_name']}"
metadata, train_data, test_data = generate_backtesting_datasets("electricity_hourly", backtest_id, total_length - 1)
history_length *= metadata.prediction_length
metadata.prediction_length *=prediction_length
mean = 0.0
std = 1.0
# chosen_rows = np.arange(321)
for entry in train_data:
    entry["target"] = (entry["target"] - mean)/std
for entry in test_data:
    entry["target"] = (entry["target"] - mean)/std
num_series = train_data[0]["target"].shape[0]

######################## Define Model ########################

model = TACTiSPTrainingNetwork(num_series=num_series, 
                               model_parameters = params["model_parameters"])

######################## Creat dataloader ########################
from torch.utils.data import DataLoader
from gluonts.dataset.field_names import FieldName
from pts.dataset.loader import TransformedIterableDataset
from gluonts.transform import (AddObservedValuesIndicator, 
                               Chain, 
                               SelectFields,
                               InstanceSampler,
                               InstanceSplitter,
                               RenameFields,
                               TestSplitSampler)
from gluonts.env import env
from gluonts.itertools import maybe_len
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
def create_instance_splitter(mode: str) :
        """
        Create and return the instance splitter needed for training, validation or testing.

        Parameters:
        -----------
        mode: str, "training", "validation", or "test"
            Whether to split the data for training, validation, or test (forecast)

        Returns
        -------
        Transformation
            The InstanceSplitter that will be applied entry-wise to datasets,
            at training, validation and inference time based on mode.
        """
        assert mode in ["training", "validation", "test"]

        if mode == "training":
            instance_sampler = SingleInstanceSampler(
                min_past=history_length,  # Will not pick incomplete sequences
                min_future=metadata.prediction_length,
            )
        elif mode == "validation":
            instance_sampler = SingleInstanceSampler(
                min_past=history_length,  # Will not pick incomplete sequences
                min_future=metadata.prediction_length,
            )
        elif mode == "test":
            # This splitter takes the last valid window from each multivariate series,
            # so any multi-window split must be done in the data definition.
            instance_sampler = TestSplitSampler()

        if False:
            normalize_transform = CDFtoGaussianTransform(
                cdf_suffix="_norm",
                target_field=FieldName.TARGET,
                target_dim=self.num_series,
                max_context_length=self.history_length,
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
                future_length=metadata.prediction_length,
                time_series_fields=[FieldName.OBSERVED_VALUES],
            )
            + normalize_transform
        )
transform = Chain(
            [
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),
            ]
        )
with env._let(max_idle_transforms=maybe_len(train_data) or 0):
            training_instance_splitter = create_instance_splitter("training")
from pts.model import get_module_forward_input_names

train_iter = TransformedIterableDataset(train_data, 
                                        transform=transform+create_instance_splitter("training")+SelectFields(['past_target_norm', 'future_target_norm']),
                                        is_train=True,
                                        shuffle_buffer_length=None,
                                        cache_data=False,

)
train_loader = DataLoader(train_iter, batch_size=params["train_batch_size"], num_workers=0, prefetch_factor=2, pin_memory=True)

######################### Distributed Essentials ############################
model_engine, optimizer, *_ = deepspeed.initialize(
    args=args, model = model, model_parameters=model.parameters()
)
SAVE_PATH = params["save_path"]
if os.path.exists(SAVE_PATH):
    _, client_ds= model_engine.load_checkpoint(SAVE_PATH)
    step = client_ds["epoch"]
else:
     step = 0

############################ datatype ###############################

dtype = torch.float
if model_engine.fp16_enabled():
     torch.set_default_dtype(torch.half)
     dtype = torch.half
elif model_engine.bfloat16_enabled():
     torch.set_default_dtype(torch.bfloat16)
     dtype = torch.bfloat16

############################ Train Loop ###############################
for epoch_no in range(step, params["epochs"]):
    # mark epoch start time
    cumm_epoch_loss = 0.0
    total = params["num_batches_per_epoch"] - 1
    # training loop
    with tqdm(train_loader, total=total, colour="green", disable=not model_engine.local_rank) as it:
        for batch_no, data_entry in enumerate(train_loader, start=1):
            torch.cuda.empty_cache()
            inputs = [v.to(model_engine.local_rank, dtype=dtype) for v in data_entry.values()]
            output = model_engine(*inputs)

            if isinstance(output, (list, tuple)):
                loss = output[0]
            else:
                loss = output

            cumm_epoch_loss += loss.item()
            avg_epoch_loss = cumm_epoch_loss / batch_no
            it.set_postfix(
                {
                    "epoch": f"{epoch_no + 1}/{params['epochs']}",
                    "avg_loss": avg_epoch_loss,
                },
                refresh=False,
            )

            model_engine.backward(loss)
            
            # if self.clip_gradient is not None:
            #     nn.utils.clip_grad_norm_(net.parameters(), self.clip_gradient)
            # print(output)
            model_engine.step()
            if batch_no==total:
                 break
            it.update(1)
        it.close()
    # if epoch_no + 1 % 2 ==0:
    model_engine.save_checkpoint(SAVE_PATH, f"ep{epoch_no+1}", client_state={"epoch":epoch_no+1})
         
    # save checkpoint
    # if (epoch_no!=0) and (epoch_no % args.save_interval == 0):
    #      model_engine.save_checkpoint
############################ Train Loop ###############################

######################### Model Evaluation ############################
# def create_predictor(transformation, trained_network):
#     pred_network = TACTiSPPredictionNetwork(
#         num_series=num_series,
#         model_parameters=trained_network
#     )
#     output_transform = cdf_to_gaussian_forward_transform if self.cdf_normalization else None
#     input_names = get_module_forward_input_names(prediction_network)
#     prediction_splitter = self.create_instance_splitter("test")
#     return PyTorchPredictor(
#         input_transform=
#     )
# TACTiSPPredictionNetwork(num_series=num_series,
#                          model_parameters=model_engine.parameters)
######################### Model Evaluation ############################
