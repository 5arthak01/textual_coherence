# from itertools import product
# from numpy import arange

# # from prepare_data import *
# # import os
# # import requests
# # import tarfile

# # print("Downloading infersent pre-trained models...")
# # os.system(
# #     "curl -Lo data/infersent1.pkl https://dl.fbaipublicfiles.com/senteval/infersent/infersent1.pkl"
# # )


# input_dropout = list(arange(0.5, 0.71, 0.1))
# hidden_layers = list(arange(1, 3))
# hidden_dropout = list(arange(0.2, 0.41, 0.1))
# margin = list(arange(4, 6.1, 1))
# l2_reg_lambda = list(arange(0, 0.11, 0.1))
# dpout_model = list(arange(0, 0.11, 0.05))
# task = ["insertion"]

# print(
#     len(input_dropout),
#     len(hidden_layers),
#     len(hidden_dropout),
#     len(margin),
#     len(dpout_model),
#     len(task),
#     len(l2_reg_lambda),
# )

# num = (
#     len(input_dropout)
#     * len(hidden_layers)
#     * len(hidden_dropout)
#     * len(margin)
#     * len(dpout_model)
#     * len(task)
#     * len(l2_reg_lambda)
# )
# print(num)
# print(f"Hours:{num/6}, Days:{num/(6*24)}")

{
    "hparams": {
        "loss": "margin",
        "input_dropout": 0.5,
        "hidden_state": 500,
        "hidden_layers": 1,
        "hidden_dropout": 0.2,
        "num_epochs": 50,
        "margin": 4.0,
        "lr": 0.001,
        "l2_reg_lambda": 0.0,
        "use_bn": False,
        "task": "discrimination",
        "bidirectional": False,
        "dpout_model": 0.0,
        "epochs": 10,
    },
    "discrimination": [
        0.9264231499051233,
        [0.8845188284518829, 0.9259776536312849, 0.9770780856423174],
    ],
    "insertion": [
        0.3061897087152976,
        [0.48480025901574014, 0.2102870837635267, 0.1343779374879815],
    ],
}
