import re
from datasets import Dataset


def load_text(path):
    with open(path, encoding='utf-8') as f:
        raw_data = f.read()
    return raw_data


def make_text_chunks(text):
    # Split by sentences using regex
    text_chunks = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    return text_chunks


def build_dataset_from_chunks(text_chunks):
    # Create a list of dictionaries from the sentences
    chungliao_data = [{"text": sentence} for sentence in text_chunks]
    chungliao_dataset = Dataset.from_list(chungliao_data)

    chungliao_dataset = chungliao_dataset.train_test_split(test_size=0.2, seed=42)

    return chungliao_dataset


def build_dataset_from_path(path):
    raw_text = load_text(path)
    raw_text = raw_text.replace('/', ' ') # remove forward slashes
    raw_text = ' '.join(raw_text.split()) # remove extra spaces tabs and new lines
    text_chunks = make_text_chunks(raw_text)
    """# filter sentences that have more than 3 words and don't contain email adresses or urls"""
    filtered_sentences = [s for s in text_chunks if len(s.split()) > 3 and not re.search(r'\S+@\S+|http\S+|www\S+|https\S+', s)] # filter sentences that have more than 3 words and don't contain
    final_data_set = build_dataset_from_chunks(filtered_sentences)
    return final_data_set
