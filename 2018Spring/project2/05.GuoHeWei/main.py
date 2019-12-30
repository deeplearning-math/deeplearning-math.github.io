from classifier import *
from dataloader import Dataloader
from config import cfg

dataloader = Dataloader(cfg.rnn.hist_size)
model = RNN(dataloader, modeltype='lstm')
# model.train()
model.validation()
model.test()
