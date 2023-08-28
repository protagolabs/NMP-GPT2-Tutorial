# NMP-GPT2-Tutorial



Please follow the format whcih we show in the `.py` file.


You can try them in the Colab, but remember when you want to upload the `.py` and `requirements.txt` file, please remove the line which will make error in `.py` fole.
> For example:
> `!pip install ...`
> `!git clone ...`

## Files

1. `stroy_customer_trainer.py`: This file, we define teh train() function to do the training.
2. `story_hf_trainer_ddp.py`:  We use the `transformers`'s `Trainer` to do the training.
3. `story_hf_trainer_model_parallel.py`: We use the `transformers`'s `Trainer` and model parallel.
