import csv
from datasets import Dataset, DatasetDict
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np



class ProbingModel(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(ProbingModel, self).__init__()
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_state):
        return self.linear(hidden_state)

class MLPProbingModel(nn.Module):
    def __init__(self, hidden_size, num_classes, n_layers):
        super(MLPProbingModel, self).__init__()
        layers = []
        input_size = hidden_size

        for _ in range(n_layers):
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size

        layers.append(nn.Linear(hidden_size, num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, hidden_state):
        return self.layers(hidden_state)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    N = predictions.shape[0]
    accuracy = (labels == predictions).sum() / N
    return {"accuracy": accuracy}

def train_probe(probe, model, train_loader, dev_loader, loss_fn, optimizer, device, num_epochs, hidden_state_layer_index):
    model = model.to(device)
    probe = probe.to(device)

    # remove pre-trained-model params from backprop
    for param in model.parameters():
        param.requires_grad = False

    results = {'train_losses': [], 'val_losses': [], 'val_accuracies': []}

    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            model_output_hidden_state = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states # get model hidden states
            model_output_hidden_state = model_output_hidden_state[hidden_state_layer_index][:, 0, :] # extract [CLS] representation at hidden state layer index

            probe_output = probe(model_output_hidden_state)

            optimizer.zero_grad()
            loss = loss_fn(probe_output, labels)
            loss.backward()
            optimizer.step()

            # Remove batch from GPU memory
            del input_ids, attention_mask, labels
            torch.cuda.empty_cache()

        results['train_losses'].append(loss.item())
        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {loss.item()}')

        # Validation
        model.eval()
        dev_loss = 0
        correct_predictions = 0
        total_predictions = 0
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc=f"Validation Epoch {epoch + 1}/{num_epochs}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                model_output_hidden_state = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states # get model hidden states
                model_output_hidden_state = model_output_hidden_state[hidden_state_layer_index][:, 0, :] # extract [CLS] representation at hidden state layer index

                probe_output = probe(model_output_hidden_state)

                loss = loss_fn(probe_output, labels)
                dev_loss += loss.item()

                predictions = probe_output.argmax(dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)

                # Remove batch from GPU memory
                del input_ids, attention_mask, labels
                torch.cuda.empty_cache()

        avg_dev_loss = dev_loss / len(dev_loader)
        accuracy = correct_predictions / total_predictions

        results['val_losses'].append(avg_dev_loss)
        results['val_accuracies'].append(accuracy)

        print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_dev_loss}, Validation Accuracy: {accuracy:.4f}')

    return results

def csv_to_dataset(path, do_train_test_split=True):
    data = []
    with open(path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Replace 'positive' with 1 and 'negative' with 0
            row['Sentiment'] = 1 if row['Sentiment'] == 'POSITIVE' else 0
            data.append(row)

    # Convert the list of dictionaries to a Hugging Face Dataset
    dataset = Dataset.from_dict({key: [d[key] for d in data] for key in data[0]})

    # Rename columns
    dataset = dataset.rename_column("Sentiment", "labels")
    dataset = dataset.rename_column("Text", "text")

    # Train test split
    if do_train_test_split:
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
        # rename splits test -> val
        dataset = DatasetDict({
                    "train": dataset["train"],
                    "val": dataset["test"]})


    return dataset

