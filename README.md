# Prepare your environment
To create virtual environment under `/scratch/[Your UID]`, `cd` to the directory and run `python -m venv [virtual environment name]`. To activate, run `source /scratch/[Your UID]/[virtual environment name]/bin/activate`. To deactivate, run `deactivate`.
After your activate your virtual environment, run `pip install -r requirements.txt` to install all the required packages.

# Training
To start training, run the following command:
```python
deepspeed [Your script name].py --deepspeed --deepspeed_config [Your deepspeed config file].json -f [Your model config file].json > [Your ckpt path].output.log &
```
- train_script: store main training script
- train_ckpt: Store checkpoint files
- train_config: Config files for training
  - `[dataset_ds_config.json]`: is the config file for `Deepspeed` engine. In our case, pls only change the `train_batch_size` and `gradient_accumulation_steps` and keep the rest unchanged.
  - `[dataset]_model_config.json` is the config for `model parameters`, total number of `epochs`, `training_batch_size` of individual GPU and `num_batches_per_epoch`.

`train_batch_size` in `~ds_config.json` is the TRUE batch size we use (the one you use with single GPU). Due to limitation of GPU memory we have to divide it into chunks. The calculation should follow the following formula:
train_batch_size[ds] = train_batch_size[model] * gradient_accumulation_steps[ds] * num_of_gpu
In my sample config for `electricity` it is `48 = 4 * 3 * 4`.
The `num_batches_per_epoch` also need to be calculated accordingly. For example, I want to train 512 batches per epoch:
num_batches_per_epoch[model] = true_batches_per_epoch * gradient_accumulation_steps[ds]
So in `~model_config.json` it is `1536 = 512 * 3`.

# Eval
You will find some `.pt` files, a `latest` and a `zero_to_fp32.py` in `train_ckpt` folder. The `.pt` cannot be used directly for evaluation. Before evaluate the model, you should convert them from deepspeed checkpoint format to a general one. To do so, run the following command, `{}` is optional:
```python
python zero_to_fp32.py {-tag [tag name]} [ckpt path] [model name].bin
```

For example:
```
python ./train_ckpt/electricity/5-8/zero_to_fp32.py ./train_ckpt/electricity/5-8 epoch_100.bin
```
If you do not specify the ckpt folder (they call it tag), by default it will use the `latest` one.

Now you have the `.bin` file, see `metrics.ipynb` for usage.