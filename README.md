# NMP-GPT2-Tutorial

This tutorial contains some code examples which showcase how you can implement our proprietary `NetmindMixins` library within your own code in order to automatically distribute the training of your code across multiple GPUs when using our NetMind Power platform.
We present three different examples, which we make available both as Google Colab notebooks, and python files.  
If you have any questions, or you are not able to apply these examples to your own code correctly, contact us at hello@netmind.ai and we'll get back to you.

## Files

1. `story_custom_trainer.py`: In this file, we show how to apply our `NetmindMixins` wrapper to a custom training function.
2. `story_hf_trainer_data_parallel.py`:  In this file we use the `transformers`'s `Trainer` to do the training and use our `NetmindMixins` wrapper to implement data parallelism.
3. `story_hf_trainer_model_parallel.py`: In this file we use the `transformers`'s `Trainer` to do the training and use our `NetmindMixins` wrapper to implement model parallelism.

## Google Colab Links

| Type | Python File Name | Colab |
|:---:|:---:|:---:|
|Custom Trainer| `story_customer_trainer.py` |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1r1WTCnvZ81du3b9WJy-1AkmFAtUOFm6J?usp=sharing) |
|HuggingFace Trainer| `story_hf_trainer.py` |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rf-0AcbQTbrb0cIp0EKzjFIeKvxXx6hh?usp=sharing) |
|Model Parallel| `story_hf_trainer_model_parallel.py` |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1VZFUlzIqd7lboJ1kGbTc3zqmo0zBB5pe?usp=sharing) |\


NOTE: If you convert a Google Colab notebook to a python file, remember to remove library install lines such as

> `!pip install ...`  
> `!git clone ...`

and anything else which is not valid python code.

