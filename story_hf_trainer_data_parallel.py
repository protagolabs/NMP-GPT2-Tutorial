# -*- coding: utf-8 -*-
"""story_hf_trainer.ipynb

Copied from story_hf_trainer.ipynb with minor editing.

Original file is located at
    https://colab.research.google.com/drive/1tUD4457MkSZZKUpsGwuA1-mGTySIJ6rV

## Introduction:


## Some infomation about this task:

About the task which we will show here is story generation.

1. Story generation: We will use the GPT-2 to train a model which can generate some stories.
2. Dataset: In huggingface "KATANABRAVE/stories"
3. [GPT model](https://huggingface.co/docs/transformers/v4.32.0/en/model_doc/gpt2#transformers.GPT2Model), we will use the model via huggingface.

Before run this notebook, please ensure that these packages you have already installed.

Packages:
numpy pandas torch torchvision torch-optimizer tqdm accelerate transformers matplotlib datasets huggingface-hub sentencepiece argparse tensorboard

**If not**, please run these codes to install all the package whcih we need. And if you have more packages whcih you want to usem. Please add them in the requirements.txt. When you upload the project, please upload the requirements.txt which you modified.
"""
import os
import transformers
from transformers import TrainingArguments, GPT2Tokenizer, AutoModelForCausalLM
from transformers import get_linear_schedule_with_warmup, AdamW, Trainer
from datasets import load_dataset
from transformers import DataCollator
from NetmindMixins.Netmind import nmp, NetmindTrainerCallback

### Step 1: Load the model and tokenizer


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.train()

### Step 2: Prepare the dataset
# Import the dataset, which is a demo for some D&D stories.
dataset = load_dataset("KATANABRAVE/stories")

### Step 3: Define the TrainingArguments

# Here we want to close the wandb, if we use the huggingface's tranier. Our Platform would allow you to add wandb later.
os.system("wandb offline")

training_args = TrainingArguments(
    f"saved_model",
    seed = 32,
    #evaluation_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    gradient_accumulation_steps=1,
    max_grad_norm=1,
    logging_steps=100,
    save_total_limit=3,
    warmup_steps=200,
    num_train_epochs=1000,
    max_steps=1000,
    save_steps=500,
    fp16=False,
    report_to="none",
)

### Step 4: Define the optimizer and scheduler.

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': training_args.weight_decay},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
optimizer = AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)

schedule_total = training_args.max_steps 

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=training_args.warmup_steps, num_training_steps=schedule_total
)

### Step 5: Initialize the Netmind nmp

nmp.init(use_ddp=True)

### Step 6: Define the NetmindTrainerCallback. We will use it in the trainer initialization

class CustomTrainerCallback(NetmindTrainerCallback):
    def __init__(self):
        super().__init__()

    '''
    Add custom training metrics
    '''

    def on_step_end(self, args: transformers.TrainingArguments, state: transformers.TrainerState,
                    control: transformers.TrainerControl, **kwargs):
        kwargs["custom_metrics"] = {}
        return super().on_step_end(args, state, control, **kwargs)

    '''
    Add custom evaluation metrics
    '''

    def on_evaluate(self, args: transformers.TrainingArguments, state: transformers.TrainerState,
                    control: transformers.TrainerControl, **kwargs):
        kwargs["custom_metrics"] = {}
        return super().on_evaluate(args, state, control, **kwargs)


### Setp 7: Start Training

nmp.init_train_bar(max_steps=training_args.max_steps)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    optimizers=(optimizer, scheduler),
    callbacks=[CustomTrainerCallback]
)
trainer.remove_callback(transformers.trainer_callback.PrinterCallback)
trainer.remove_callback(transformers.trainer_callback.ProgressCallback)

trainer.train()

nmp.finish_training() # Finish the training. It should be placed at the end of file
