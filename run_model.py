from models.coherence_models import BigramCoherence
from preprocess import (
    get_sbert,
    get_average_glove,
    save_eval_perm,
    get_t5,
)
from utils.data_utils import DataSet
from utils.logging_utils import _set_basic_logging
import logging
import config
from torch.utils.data import DataLoader
import os
import argparse
from add_args import add_bigram_args
import torch
from datetime import datetime

from json import dump
import pickle


def print_current_time():
    print("\n\nThe time is: {}".format(datetime.now().isoformat()))


SAVED_GLOVE = False
SAVED_SBERT = False
PRETRAINED_MODEL = False


def run_bigram_coherence(args):
    logging.info("Loading data...")
    if args.data_name not in config.DATASET:
        raise ValueError("Invalid data name!")
    dataset = DataSet(config.DATASET[args.data_name])
    if not os.path.isfile(dataset.test_perm):
        save_eval_perm(args.data_name, random_seed=args.random_seed)

    train_dataset = dataset.load_train(args.portion)
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    valid_dataset = dataset.load_valid(args.portion)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False)
    valid_df = dataset.load_valid_perm()

    test_dataset = dataset.load_test(args.portion)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    test_df = dataset.load_test_perm()

    logging.info("Loading sent embedding...")
    if args.sent_encoder == "sbert":
        if not SAVED_SBERT:
            sent_embedding = get_sbert(args.data_name, if_sample=args.test)
            with open("./data/sbert.pkl", "wb") as f:
                pickle.dump(sent_embedding, f)
        else:
            with open("./data/sbert.pkl", "rb") as f:
                sent_embedding = pickle.load(f)
        embed_dim = 768
    elif args.sent_encoder == "average_glove":
        if not SAVED_GLOVE:
            sent_embedding = get_average_glove(args.data_name, if_sample=args.test)
            with open("./data/glove.pkl", "wb") as f:
                pickle.dump(sent_embedding, f)
        else:
            with open("./data/glove.pkl", "rb") as f:
                sent_embedding = pickle.load(f)
        embed_dim = 300
    elif args.sent_encoder == "t5":
        sent_embedding = get_t5(args.data_name, if_sample=args.test)
        embed_dim = 512
    else:
        raise ValueError("Invalid sent encoder name!")

    name_key = "t5"

    RESULTS_DIR = f"{config.ROOT_PATH}/results_" + name_key
    MODEL_SAVE_PATH = RESULTS_DIR + "/models"
    RESULTS_SAVE_PATH = RESULTS_DIR
    os.makedirs(RESULTS_SAVE_PATH, exist_ok=True)
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    # Hyperparameters
    kwargs = {
        "embed_dim": embed_dim,
        "sent_encoder": sent_embedding,
        "hparams": {
            "loss": "margin",
            "input_dropout": 0.5,
            "hidden_state": 500,
            "hidden_layers": 1,
            "hidden_dropout": 0.3,
            "num_epochs": 50,
            "margin": 5.0,
            "lr": 0.001,
            "l2_reg_lambda": 0.0,
            "use_bn": False,
            "task": "discrimination",
            "bidirectional": False,
            "scorer": None,
        },
    }

    logging.info("Training BigramCoherence model...")
    if PRETRAINED_MODEL:
        model = torch.load(f"{MODEL_SAVE_PATH}/bigram_coherence_model.pt")
    else:
        model = BigramCoherence(**kwargs)
        model.init()
        best_step, valid_acc = model.fit(train_dataloader, valid_dataloader, valid_df)
        torch.save(model, f"{MODEL_SAVE_PATH}/bigram_coherence_model.pt")

    model.load_best_state()

    logging.info("Results for discrimination:")
    dis_acc, _ = model.evaluate_dis(test_dataloader, test_df)
    logging.info("Discr Test Acc:", dis_acc)

    logging.info("Results for insertion:")
    ins_acc, _ = model.evaluate_ins(test_dataloader, test_df)
    logging.info("Ins Test Acc:", ins_acc)

    # Save results
    results_path = os.path.join(
        RESULTS_SAVE_PATH,
        f"{name_key}-{valid_acc:.4f}",
    )
    results = {
        "hparams": kwargs["hparams"],
        "discrimination": dis_acc,
        "insertion": ins_acc,
    }

    with open(results_path + ".json", "w") as f:
        dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_bigram_args(parser)
    args = parser.parse_args()

    _set_basic_logging()
    run_bigram_coherence(args)
