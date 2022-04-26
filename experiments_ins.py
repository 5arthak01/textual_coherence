from models.coherence_models import BigramCoherence
from preprocess import get_infersent, get_average_glove, save_eval_perm, get_lm_hidden
from preprocess import get_s2s_hidden
from utils.data_utils import DataSet
from utils.lm_utils import Corpus, SentCorpus
from utils.logging_utils import _set_basic_logging
import logging
import config
from torch.utils.data import DataLoader
import os
import argparse
from add_args import add_bigram_args
import torch
from datetime import datetime
from numpy import arange
from itertools import product
from json import dump


def print_current_time():
    print("\n\nThe time is: {}".format(datetime.now().isoformat()))


def run_bigram_coherence(args):
    # logging.info("Loading data...")
    if args.data_name not in config.DATASET:
        raise ValueError("Invalid data name!")
    dataset = DataSet(config.DATASET[args.data_name])
    # dataset.random_seed = args.random_seed
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

    # logging.info("Loading sent embedding...")
    if args.sent_encoder == "infersent":
        sent_embedding = get_infersent(args.data_name, if_sample=args.test)
        embed_dim = 4096
    elif args.sent_encoder == "average_glove":
        sent_embedding = get_average_glove(args.data_name, if_sample=args.test)
        embed_dim = 300
    elif args.sent_encoder == "lm_hidden":
        corpus = Corpus(train_dataset.file_list, test_dataset.file_list)
        sent_embedding = get_lm_hidden(args.data_name, "lm_" + args.data_name, corpus)
        embed_dim = 2048
    elif args.sent_encoder == "s2s_hidden":
        corpus = SentCorpus(train_dataset.file_list, test_dataset.file_list)
        sent_embedding = get_s2s_hidden(args.data_name, "s2s_" + args.data_name, corpus)
        embed_dim = 2048
    else:
        raise ValueError("Invalid sent encoder name!")

    # logging.info("Training BigramCoherence model...")
    # print("Training BigramCoherence model...")
    kwargs = {
        "embed_dim": embed_dim,
        "sent_encoder": sent_embedding,
        "hparams": {
            "loss": args.loss,
            "input_dropout": args.input_dropout,
            "hidden_state": args.hidden_state,
            "hidden_layers": args.hidden_layers,
            "hidden_dropout": args.hidden_dropout,
            "num_epochs": args.num_epochs,
            "margin": args.margin,
            "lr": args.lr,
            "l2_reg_lambda": args.l2_reg_lambda,
            "use_bn": args.use_bn,
            "task": "discrimination",
            "bidirectional": args.bidirectional,
        },
    }

    input_dropout = list(arange(0.5, 0.71, 0.1))
    hidden_layers = list(arange(1, 3))
    hidden_dropout = list(arange(0.2, 0.41, 0.1))
    margin = list(arange(4, 6.1, 1))
    l2_reg_lambda = list(arange(0, 0.11, 0.1))
    dpout_model = list(arange(0, 0.11, 0.05))
    task = ["insertion"]
    # os.makedirs(config.CHECKPOINT_PATH, exist_ok=True)

    RESULTS_PATH = "%s/results_%s" % config.ROOT_PATH, task[0]
    os.makedirs(config.RESULTS_PATH, exist_ok=True)
    all_results = []

    for i, x in enumerate(
        product(
            input_dropout,
            hidden_layers,
            hidden_dropout,
            margin,
            l2_reg_lambda,
            dpout_model,
            task,
        )
    ):
        kwargs["hparams"]["input_dropout"] = x[0]
        kwargs["hparams"]["hidden_layers"] = x[1]
        kwargs["hparams"]["hidden_dropout"] = x[2]
        kwargs["hparams"]["margin"] = x[3]
        kwargs["hparams"]["l2_reg_lambda"] = x[4]
        kwargs["hparams"]["dpout_model"] = x[5]
        kwargs["hparams"]["task"] = x[6]

        print_current_time()
        model = BigramCoherence(**kwargs)
        model.init()
        best_step, valid_acc = model.fit(train_dataloader, valid_dataloader, valid_df)

        # Save model
        model_path = os.path.join(RESULTS_PATH, "%06d-%.4f" % (i, valid_acc))
        torch.save(model, model_path + ".pth")
        model.load_best_state()

        print_current_time()
        print("Results for discrimination:")
        dis_acc = model.evaluate_dis(test_dataloader, test_df)
        print("Test Acc:", dis_acc)

        print_current_time()
        print("Results for insertion:")
        ins_acc = model.evaluate_ins(test_dataloader, test_df)
        print("Test Acc:", ins_acc)

        # Save results
        results_path = os.path.join(RESULTS_PATH, "%06d-%.4f" % (i, valid_acc))
        results = {"kwargs": kwargs, "discrimination": dis_acc, "insertion": ins_acc}
        with open(results_path + ".json", "w") as f:
            dump(results, f, indent=4)
        all_results.append(results)

    with open(RESULTS_PATH + "all_results" + ".json", "w") as f:
        dump(all_results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_bigram_args(parser)
    args = parser.parse_args()

    _set_basic_logging()
    run_bigram_coherence(args)
