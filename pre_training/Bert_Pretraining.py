from build_dataset import build_dataset_from_path
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForMaskedLM
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
import torch
import os.path
import sys
import argparse


parser=argparse.ArgumentParser()

parser.add_argument("--learning_rate")
parser.add_argument("--epochs")
parser.add_argument("--output_dir")
parser.add_argument("--batch_size")
parser.add_argument("--weight_decay")
parser.add_argument("--warmup_ratio")
parser.add_argument("--model_name_or_path")
parser.add_argument("--dataset_path")
args=parser.parse_args()

# default pre-trained model
model_name_or_path = "google-bert/bert-base-multilingual-cased"
# default output dir
output_dir = "models"


# Hyperparams
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 20
WEIGHT_DECAY = 0.01
WARM_UP_RATIO = 0.1

if args.batch_size:
    BATCH_SIZE = args.batch_size
if args.learning_rate:
    LEARNING_RATE = args.learning_rate
if args.epochs:
    EPOCHS = args.epochs
if args.warmup_ratio:
    WARM_UP_RATIO = args.warmup_ratio
if args.model_name_or_path:
    model_name_or_path = args.model_name_or_path
if args.output_dir:
    output_dir = args.output_dir



print("torch version: ", torch.__version__)
if torch.cuda.is_available():
    print("CUDA is available. PyTorch is using GPU.")
else:
    print("CUDA is not available. PyTorch is using CPU.")
print("\n")

full_path = args.dataset_path

print("loading data from: ", full_path)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)


ChungliAo_dataset = build_dataset_from_path(full_path)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding=True,
                                          truncation=True)

tokenized_ChungliAo_dataset = ChungliAo_dataset.map(
    preprocess_function,
    batched=True,
)

def get_gradient_accumulation_steps(batch_size):
    """
    :param batch_size: batch size that is below 8 or divisible by 8
    :type batch_size:
    :return:
    :rtype:
    """
    if batch_size <= 8:
        return 1
    return batch_size // 8

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

GRADIENT_ACCUMULATION_STEPS = get_gradient_accumulation_steps(BATCH_SIZE)
print("Batch Size: ", BATCH_SIZE, "gradient_accumulation_steps: ", GRADIENT_ACCUMULATION_STEPS,
          "per_device_train_batch_size: ", BATCH_SIZE // GRADIENT_ACCUMULATION_STEPS)


model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)

training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=5,
    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARM_UP_RATIO,
    per_device_train_batch_size=BATCH_SIZE // GRADIENT_ACCUMULATION_STEPS,
    # set correct batch size for gradient accumulation steps
    # (e.g. BatchSize=32 // 4 = 8, BatchSize=64 // 8 = 8)
    per_device_eval_batch_size=BATCH_SIZE // GRADIENT_ACCUMULATION_STEPS,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ChungliAo_dataset["train"],
    eval_dataset=tokenized_ChungliAo_dataset["test"],
    data_collator=data_collator,
)

trainer.train()
