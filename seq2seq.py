from fastai.text.all import *
from utils import *


def seq2seq_loss(out, targ, pad_idx=1):
    bs, targ_len = targ.size()
    _, out_len, vs = out.size()
    if targ_len > out_len: out = F.pad(out, (0, 0, 0, targ_len - out_len, 0, 0), value=pad_idx)
    if out_len > targ_len: targ = F.pad(targ, (0, out_len - targ_len, 0, 0), value=pad_idx)
    return CrossEntropyLossFlat()(out, targ)


def seq2seq_acc(out, targ, pad_idx=1):
    bs, targ_len = targ.size()
    _, out_len, vs = out.size()
    if targ_len > out_len: out = F.pad(out, (0, 0, 0, targ_len - out_len, 0, 0), value=pad_idx)
    if out_len > targ_len: targ = F.pad(targ, (0, out_len - targ_len, 0, 0), value=pad_idx)
    out = out.argmax(2)
    return (out == targ).float().mean()


class GetPreds:
    def __init__(self, dls, inputs, preds, targs):
        self.dls, self.inputs, self.preds, self.targs = dls, inputs, preds, targs

    def get_predictions(self, num, ignore_pad=False): 
        """:ignore_pad: Whether to ignore pad for predictions. Default: False"""
        return (
            itos(self.dls.vocab[0], self.inputs[num], join=True, ignore_pad=True),
            itos(self.dls.vocab[1], self.targs[num], join=True, ignore_pad=True),
            itos(self.dls.vocab[1], self.preds[num].argmax(1), join=True, ignore_pad=ignore_pad)
        )


class TeacherForcing(Callback):
    def __init__(self, end_epoch, full_force_for=0): 
        """
        :full_force_for: Training with 1.0 teacher forcing (full force)
            for how many epochs? One found this useful for maintaining
            full teacher forcing at the beginning (encourage to use fff = 3)
            might make model learn better. Further research is required
            on whether this is useful across different corpus, models, and
            domains. Default: 0 (full force only for first epoch). 
        """

        self.fff = full_force_for - 1  # start counting from zero. 
        self.end_epoch = end_epoch
    
    def before_batch(self): 
        self.learn.xb = (self.x, self.y)

    def before_epoch(self):
        self.learn.model.pr_force = 1 - ((self.learn.epoch - self.fff) / (self.end_epoch - self.fff))
        if self.learn.epoch <= self.fff: self.learn.model.pr_force = 1