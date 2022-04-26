from itertools import product
from numpy import arange

# from prepare_data import *
# import os
# import requests
# import tarfile

# print("Downloading infersent pre-trained models...")
# os.system(
#     "curl -Lo data/infersent1.pkl https://dl.fbaipublicfiles.com/senteval/infersent/infersent1.pkl"
# )

# loss = ["margin", "log", "margin+log"]
input_dropout = list(arange(0.5, 0.71, 0.1))
# hidden_state = list(arange(300, 701, 100))
hidden_layers = list(arange(1, 3))
hidden_dropout = list(arange(0.2, 0.41, 0.1))
# num_epochs = list(arange(40, 61, 10))
margin = list(arange(4, 6.1, 1))
l2_reg_lambda = list(arange(0, 0.11, 0.1))
# lr = list(arange(0.001, 0.0021, 0.001))
dpout_model = list(arange(0, 0.11, 0.05))
task = ["discrimination", "insertion"]

print(
    len(input_dropout),
    len(hidden_layers),
    len(hidden_dropout),
    len(margin),
    len(dpout_model),
    len(task),
    len(l2_reg_lambda),
)

num = (
    # len(loss)
    len(input_dropout)
    # * len(hidden_state)
    * len(hidden_layers)
    * len(hidden_dropout)
    # * len(num_epochs)
    * len(margin)
    # * len(lr)
    * len(dpout_model)
    * len(task)
    * len(l2_reg_lambda)
)

print(num / (6 * 24))
