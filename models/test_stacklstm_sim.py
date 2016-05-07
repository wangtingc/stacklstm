from stack_lstm_sim import StackLSTMSim
from lasagne.init import Normal

def test():
    sls = StackLSTMSim(512, 80, 2, 10000, 300, Normal(), 0.5, 0.5)
    f_train = sls.get_f_train()

test()
