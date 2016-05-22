from stack_lstm_lm import StackLSTMLM
from lasagne.init import Normal
import numpy as np

def test():
    slm = StackLSTMLM(512, 20000, 300, np.zeros([20000,300]), 0.2)
    f_train = slm.get_f_train()

test()
