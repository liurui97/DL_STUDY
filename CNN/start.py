import matplotlib.pyplot as plt
from cnn import *
from data_utils import get_CIFAR10_data
from solver import Solver

data = get_CIFAR10_data()
model = ThreeLayerConvNet(reg=0.9)
solver = Solver(model, data,
                lr_decay=0.95,
                print_every=100, num_epochs=40, batch_size=400,
                update_rule='sgd_momentum',
                optim_config={'learning_rate': 5e-4, 'momentum': 0.9})