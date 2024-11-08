from collections import defaultdict
from warnings import filterwarnings
filterwarnings('ignore')

import torch
import datasets
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline

from models import BottleneckT5Autoencoder
from .base_dataset import BasePreferenceDataset


PROMPT_LENGTH = 5
MAX_TOKEN_LENGTH = 20
IMDB_LABEL_MAP = {
    0: 'NEGATIVE',
    1: 'POSITIVE',
}


class IMDBEvaluator(object):
    
    _sentiment_pipe_kwargs = {"top_k": None, "function_to_apply": "none"}
    
    def __init__(self, checkpoint="lvwerra/distilbert-imdb", device='cuda'):
        self.device = device
        self.pipe = pipeline("sentiment-analysis", checkpoint, device=device)
    
    def __call__(self, text: str, label: str="POSITIVE"):
        
        if type(label) is int:
            label = IMDB_LABEL_MAP[label]
        
        scores = self.pipe(text, **self._sentiment_pipe_kwargs)
        for element in scores:
            if element['label'] == label:
                return element["score"]
            
        raise ValueError(f"Invalid label {label}")


def load_imdb_dataset(
    split: str='train',
    classifier_checkpoint: str="lvwerra/distilbert-imdb",
    pretrained_checkpoint: str="lvwerra/gpt2-imdb",
    autoencoder_checkpoint: str='thesephist/contra-bottleneck-t5-large-wikipedia',
    prompt_length: int=PROMPT_LENGTH,
    max_token_length: int=MAX_TOKEN_LENGTH,
    device: torch.device='cuda',
    ) -> torch.utils.data.Dataset:
    """
    Loads IMDB preference dataset.
    """
    # load the huggingface dataset and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_checkpoint)
    dataset = datasets.load_dataset('imdb', split=split, trust_remote_code=True)
    dataset = dataset.filter(lambda x: len(x["text"]) > 100, batched=False) # filter data that are too short
    dataset = dataset.map(lambda x: {"text": x["text"][:500]}, batched=False) # early trimming for faster tokenization
    
    # tokenize the texts and trim to max_token_length.
    dataset = dataset.map(
        lambda x: {
            "query": tokenizer.encode(" " + x["text"], return_tensors="pt")[0, :prompt_length],
            "text": tokenizer.encode(" " + x["text"], return_tensors="pt")[0, prompt_length:max_token_length + prompt_length],
        },
        batched=False,
    )

    # decode the tokens to obtain trimmed texts.
    dataset = dataset.map(
        lambda x: {
            "query": tokenizer.decode(x["query"]),
            "text": tokenizer.decode(x['text']),
        }, 
        batched=False
    )
    
    # preference evaluator for dataset processing
    evaluator = IMDBEvaluator(
        checkpoint=classifier_checkpoint, 
        device=device
    )
    
    # iterate over the raw dataset to obtain preference dataset
    scored_dataset = defaultdict(list)
    autoencoder = BottleneckT5Autoencoder(
        model_path=autoencoder_checkpoint, 
        device=device
    )
    
    for data in tqdm(dataset, desc="Processing dataset"):
        label, text = data['label'], data['text']
        
        score = evaluator(text, label=label)
        embedding = autoencoder.encode(text).detach().cpu()
        embedding = embedding.view(32, 32)
        scored_dataset[label].append((embedding, score))
        
    # iterate over each label and sort them according to scores
    for label in scored_dataset:
        data = scored_dataset[label]
        data = sorted(data, key=lambda x: x[1])
        scored_dataset[label] = [d[0] for d in data]
        
    # parse the dataset into desired PFM format
    positive = []
    negative = []
    
    for label in scored_dataset:
        data = scored_dataset[label]
        mid_point = len(data) // 2
        for i in range(mid_point):
            positive.append(data[i + mid_point])
            negative.append(data[i])
            
    dataset = BasePreferenceDataset(positive, negative)
    return dataset


    

