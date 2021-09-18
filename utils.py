"""
Utils to ease backward dependencies between fastai v1 and v2. 
"""
from multiprocessing import Value
from fastai.layers import trunc_normal_
import numpy as np
import torch
import torch.nn as nn
import fastai


def stoi(vocab, word): 
    if not isinstance(vocab, np.ndarray): vocab = np.array(vocab)
    if type(word) in [list, tuple, np.ndarray]: 
        m = np.zeros((len(word), )).astype(np.uint8)
        for k in range(len(word)):
            try: m[k] = np.where(vocab == word[k])[0].item()
            except ValueError: raise ValueError("word not in vocab")
        return m
    try: return np.where(vocab == word)[0].item()
    except ValueError: raise ValueError("word not in vocab")

def itos(vocab, word, join=False, ignore_pad=False): 
    if not isinstance(vocab, np.ndarray): vocab = np.array(vocab)
    if type(word) in [tuple, fastai.text.data.TensorText]: 
        word = np.array(word)
    word = vocab[word]
    if ignore_pad: word = word[word != "xxpad"]
    if join: return " ".join(word)

    return word

def str_list_to_par(dls, c, train=True): 
    """Previously get_train_x_c. 
    String list to paragraph. Original, a list of strings, and concat them.
    This takes the training data and put them in."""
    if train==True: return " ".join(dls.train_ds.items.text[c])
    else: return " ".join(dls.valid_ds.items.text[c])

def str_list_to_par_wpath(dls, c, train=True):
    """
    c: the index. 
    train: whether this is a training dataset or not. 
    """
    ds = dls.train_ds if train else dls.valid_ds
    
    vocab = np.array(dls.vocab[0])
    tensor_text_np = np.array(ds[c][0])

    return " ".join(vocab[tensor_text_np])

def get_y(dls, train=True):
    """same as movie_reviews.train.y[index] (or .valid) for fastai v1. 
        Only supports if target is of single element. """
    ds = dls.train_ds if train else dls.valid_ds

    m = np.zeros((len(ds), ))
    try:
        for k in range(len(ds)): m[k] = int(ds[k][1])
    except ValueError: 
        raise NotImplementedError("Multi-element target not implemented.")

    return m


def embedding(ni: int, nf: int) -> torch.nn.modules.module.Module: 
    "Create an embedding layer"
    emb = nn.Embedding(ni, nf)
    with torch.no_grad(): trunc_normal_(emb.weight, std=0.01)
    
    return emb