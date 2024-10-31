from build_dataset import build_dataset_from_path
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForMaskedLM
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
import torch
import os.path
import wandb


wandb.login()

print("torch version: ", torch.__version__)
if torch.cuda.is_available():
    print("CUDA is available. PyTorch is using GPU.")
else:
    print("CUDA is not available. PyTorch is using CPU.")
print("\n")

path_prefix = os.path.dirname(__file__)
full_path = path_prefix + "/../data/raw_text.txt"

print("loading data from: ", full_path)


os.environ["WANDB_PROJECT"] = "ChungliAoPreTraining"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)


ChungliAo_dataset = build_dataset_from_path(full_path)

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased", padding=True,
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

# Hyperparams
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 20
WEIGHT_DECAY = 0.01
WARM_UP_RATIO = 0.1


GRADIENT_ACCUMULATION_STEPS = get_gradient_accumulation_steps(BATCH_SIZE)
print("Batch Size: ", BATCH_SIZE, "gradient_accumulation_steps: ", GRADIENT_ACCUMULATION_STEPS,
          "per_device_train_batch_size: ", BATCH_SIZE // GRADIENT_ACCUMULATION_STEPS)

model_path = "google-bert/bert-base-multilingual-cased"

model = AutoModelForMaskedLM.from_pretrained(model_path)

training_args = TrainingArguments(
    output_dir="models",
    eval_strategy="epoch",
    save_strategy="epoch",
    run_name="Chungli-Ao-MBERT-cased_final_v2",
    report_to="wandb",
    load_best_model_at_end=False,
    save_total_limit=5,
    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARM_UP_RATIO,
    per_device_train_batch_size=BATCH_SIZE // GRADIENT_ACCUMULATION_STEPS,
    # set correct batch size for gradient accumulation steps
    # (e.g. BatchSize=32 // 2 = 16, BatchSize=64 // 4 = 16)
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
