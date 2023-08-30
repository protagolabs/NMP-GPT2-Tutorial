# NMP-GPT2-Tutorial

Please follow the format whcih we show in the `.py` file.

You can try them in the Colab, but remember when you want to upload the `.py` and `requirements.txt` file, please remove the line which will make error in `.py` fole.

> For example:
> `!pip install ...`
> `!git clone ...`

## Files

1. `stroy_customer_trainer.py`: This file, we define the train() function to do the training.
2. `story_hf_trainer.py`:  We use the `transformers`'s `Trainer` to do the training.
3. `story_hf_trainer_model_parallel.py`: We use the `transformers`'s `Trainer` and model parallel.

## Codlab Link

| Type | Python File Name | Colab |
|:---:|:---:|:---:|
|Customer Trainer| `stroy_customer_trainer.py` |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1r1WTCnvZ81du3b9WJy-1AkmFAtUOFm6J?usp=sharing)] |
|HuggingFace Trainer| `story_hf_trainer.py` |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rf-0AcbQTbrb0cIp0EKzjFIeKvxXx6hh?usp=sharing)] |
|Model Parallel| `story_hf_trainer_model_parallel.py` |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1VZFUlzIqd7lboJ1kGbTc3zqmo0zBB5pe?usp=sharing))] |\

