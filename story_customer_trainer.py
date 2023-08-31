# -*- coding: utf-8 -*-
"""story_customer_trainer.ipynb

## Some infomation about this task:

1. Story generation: We will use the GPT-2 to train a model which can generate some stories.
2. Dataset: We will use the "KATANABRAVE/stories" dataset from HuggingFace
3. [GPT model](https://huggingface.co/docs/transformers/v4.32.0/en/model_doc/gpt2#transformers.GPT2Model), we will use the HuggingFace implementation

Ensure you have install the correct libraries before running this code.

Required packages:
numpy pandas torch torchvision torch-optimizer tqdm accelerate transformers matplotlib datasets huggingface-hub sentencepiece argparse tensorboard
If your modified code includes other additional libraries, please add them to the requirements.txt file before uploading 
the project to the NetMind Power platform, otherwise the platform environment may not build correctly.
"""


import torch
from tqdm import tqdm
import argparse
import os
from torch.nn.utils import clip_grad_norm_
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from NetmindMixins.Netmind import nmp, NetmindOptimizer, NetmindDistributedModel


# Step 1: Load the model and tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.train()


# Step 2: Prepare the dataset.
# Import the dataset, which is a demo for some stories.

stories = load_dataset("KATANABRAVE/stories")

train_data = stories["train"]
eval_data = stories["validation"]

train_dataloader = DataLoader(train_data, batch_size=4, shuffle=False)


# Step 3: Define the training parameters
# Custom training loop

def setup_args():
    """
    Set training parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default= 'gpt2' , type=str, required=False, help='')
    parser.add_argument('--per_device_train_batch_size', default= 4 , type=int, required=False, help='')
    parser.add_argument('--learning_rate', default= 2e-4 , type=float, required=False, help='')
    parser.add_argument('--num_train_epochs', default= 10000 , type=int, required=False, help='')

    # adv
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--max_grad_norm", default=1, type=float)
    parser.add_argument("--warmup_steps", default=500, type=float)
    parser.add_argument('--output_dir', default= 'model_1' , type=str, required=False, help='')
    parser.add_argument('--save_steps', default=500, type=int, required=False, help='')
    parser.add_argument('--max_steps', default=1000, type=int, required=False, help='')

    # distributed learning
    parser.add_argument(
        "--local_rank",
        type=int,
        default=os.getenv('LOCAL_RANK', 0),
        help="Local rank. Necessary for using the torch.distributed.launch utility"
    )

    return parser.parse_known_args()[0]


training_args = setup_args()


# Step 4: Define the optimizer
"""
NOTE: The optimizer should suit the model you are using. You should then wrap it within the NetmindOptimizer class as shown below
optimizer = NetmindOptimizer(optimizer)
"""

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': training_args.weight_decay},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
optimizer = AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)

schedule_total = training_args.max_steps


# Step 5: Define the customer trainer
# Note that we need insert step_callback for monitoring training loss.

def train(dataset, training_args, model, optimizer, scheduler, step_callback):

    schedule_total = training_args.max_steps

    train_data = dataset

    device = torch.device("cuda:{}".format(training_args.local_rank))
    completed_steps = 0
    for epoch in range(training_args.num_train_epochs):
        progress_bar = tqdm(range( training_args.max_steps ))
        progress_bar.set_description(f'**Epoch: {epoch}**')

        total_loss = 0

        for train_step, batch in enumerate(train_data):
            optimizer.zero_grad()

            input_ids = torch.tensor([ids.tolist() for ids in batch['input_ids']])
            attention_mask = torch.tensor([ids.tolist() for ids in batch['attention_mask']])
            labels = torch.tensor([ids.tolist() for ids in batch['labels']])

            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
            loss = outputs.loss
            # We keep track of the loss at each epoch

            total_loss += loss.detach().float()
            # loss = loss / self.gradient_accumulation_steps
            # accelerator.backward(loss)
            loss.backward()
            if training_args.max_grad_norm > 0:
                clip_grad_norm_(model.parameters(), training_args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            # average loss in one epoch
            loss2log = total_loss.item()/ (train_step+1)
            lr2log  = scheduler.get_last_lr()[0]
            progress_bar.set_postfix(loss=loss2log , lr=lr2log )
            progress_bar.update(1)
            completed_steps += 1

            monitor_metrics = {
                "loss": loss.item(),
                "Learning rate": scheduler.get_last_lr()[0]
            }

            step_callback(monitor_metrics)

            if completed_steps == training_args.max_steps:
                return

    # Just for nividia-smi visiable memory release
    torch.cuda.empty_cache()


# Step 6: Set the GPU

device = torch.device("cuda:{}".format(training_args.local_rank))
model.to(device)


# Step 7: Initialize the Netmind nmp

nmp.init(use_ddp=True)


# Step 8: Set the model to NetmindDistributedModel
"""
#### NetmindDistributedModel(model)
- model: Model variable.
  
Wrap the machine learning model within "NetmindDistributedModel". This will not change the ML model itself. 
It can be placed anywhere after the "model" is defined and before the actual start of training.

#### NetmindOptimizer(optimizer)
- optimizer: optimizer variable.

Wrap the optimizer within "NetmindOptimizer". This will not change the optimizer itself. 
It can be placed anywhere after the "optimizer" is defined and before the actual start of training.
"""
ddp_model = NetmindDistributedModel(
    torch.nn.parallel.DistributedDataParallel(model, device_ids=[training_args.local_rank], output_device=training_args.local_rank))

optimizer = NetmindOptimizer(optimizer)

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=training_args.warmup_steps, num_training_steps=schedule_total
)


# Setp 9: Start Training
# Set the process bar and start the training.

nmp.init_train_bar(max_steps=training_args.max_steps)
train(train_dataloader, training_args, model, optimizer, scheduler, nmp.step)
nmp.finish_training() # Finish the training. It should be placed at the end of file
