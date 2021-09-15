"""
Test files for utils.py
Created on: 07 Septembre 2021.
"""
import pytest
from fastai.text.all import *
from utils import *

path = untar_data(URLs.IMDB_SAMPLE)
df = pd.read_csv(path/"texts.csv")

imdb_clas = DataBlock(
    blocks=(TextBlock.from_df("text", seq_len=72), CategoryBlock),
    get_x=ColReader("text"), get_y=ColReader("label"), splitter=ColSplitter()
)

dls = imdb_clas.dataloaders(df, bs=32)


# ------------------------ #
def test_itos_single_vocab_works():
    assert itos(dls.vocab[0], 32) == dls.vocab[0][32]


def test_itos_multi_vocab_works():
    assert len(itos(dls.vocab[0], [17, 42, 55])) == 3

    v = dls.vocab[0]
    assert (itos(dls.vocab[0], [17, 42, 55]) == np.array([v[17], v[42], v[55]])).all()


def test_itos_multi_with_tuple_works(): 
    assert len(itos(dls.vocab[0], (17, 42, 55))) == 3

    v = dls.vocab[0]
    assert (itos(dls.vocab[0], (17, 42, 55)) == np.array([v[17], v[42], v[55]])).all()


def test_itos_multi_with_nparray_works():
    assert len(itos(dls.vocab[0], np.array([17, 42, 55]))) == 3

    v = dls.vocab[0]
    assert (itos(dls.vocab[0], np.array([17, 42, 55])) == np.array([v[17], v[42], v[55]])).all()


def test_itos_multi_with_tensortext_correct_len(): 
    assert len(itos(dls.vocab[0], dls.train_ds[0][0])) == len(dls.train_ds[0][0])


def test_itos_multi_join_works():
    assert type(itos(dls.vocab[0], dls.train_ds[0][0], join=True)) == str

def test_itos_remove_pad_works():
    example_sentence = TensorText([   2,   25, 2053,   10,   69, 1249,   20, 4076,   42,   13, 2180,    9,
           1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
           1,    1,    1,    1,    1,    1])
    
    assert "xxpad" in itos(dls.vocab[0], example_sentence)
    assert not "xxpad" in itos(dls.vocab[0], example_sentence, ignore_pad=True)


# Up till this point we assume itos works fine. 

def test_stoi_single_word_works():
    m = dls.vocab[0][61]
    assert stoi(dls.vocab[0], m) == 61

def test_stoi_multi_vocab_works():
    m = np.array(["what", "is", "this", "thing"])  # random list

    assert len(stoi(dls.vocab[0], m)) == len(m)
    assert (itos(dls.vocab[0], stoi(dls.vocab[0], m)) == m).all()

def test_stoi_support_for_tuples():
    m = ("what", "is", "this", "thing")

    assert len(stoi(dls.vocab[0], m)) == len(m)
    assert (itos(dls.vocab[0], stoi(dls.vocab[0], m)) == np.array(m)).all()

def test_stoi_support_for_python_list():
    m = ["what", "is", "this", "thing"]

    assert len(stoi(dls.vocab[0], m)) == len(m)
    assert (itos(dls.vocab[0], stoi(dls.vocab[0], m)) == np.array(m)).all()

# ----------

def test_str_list_to_par_works():
    assert dls.train_ds.items.text[37] == str_list_to_par(dls, 37).split(" ")

