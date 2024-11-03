### Chungli Ao Sentiment Analysis

## Pre-Training
Pre-Training Dataset Builder: `pre_training/build_dataset.py`  
Main Pre-Training Script: `pre_training/Bert_Pretraining.py`  

To run Pre-Training script use:  
`python pre_training/Bert_Pretraining.py --model_name_or_path [huggingface repository or path] --dataset_path Path/to/dataset --output_dir [path/to/output_dir] --epochs [] --learning_rate [] --batch_size [] --warmup_ratio [ratio of warmup stepts to total training steps]`  
Example:    
`python pre_training/Bert_Pretraining.py --model_name_or_path "google-bert/bert-base-multilingual-cased" --dataset_path "pre_training/data" --output_dir "models" --epochs 20 --learning_rate 1e-4 --batch_size 16 --warmup_ratio 0.1` 