from functools import cache
import os
import logging
import config
from utils.logging_utils import _set_basic_logging
from utils.data_utils import DataSet
from sentence_transformers import SentenceTransformer
import numpy as np
import random
import copy
import itertools
import argparse
from transformers import T5Model, T5Tokenizer


def permute_articles(cliques, num_perm):
    permuted_articles = []
    for clique in cliques:
        clique = list(clique)
        old_clique = copy.deepcopy(clique)
        random.shuffle(clique)
        perms = itertools.permutations(clique)
        inner_perm = []
        i = 0
        for perm in perms:
            comparator = [old_sent == sent for old_sent, sent in zip(old_clique, perm)]
            if not np.all(comparator):
                inner_perm.append(list(perm))
                i += 1
            if i >= num_perm:
                break
        permuted_articles.append(inner_perm)
    return permuted_articles


def permute_articles_with_replacement(cliques, num_perm):
    permuted_articles = []
    for clique in cliques:
        clique = list(clique)
        old_clique = copy.deepcopy(clique)
        inner_perm = []
        i = 0
        while i < num_perm:
            random_perm = copy.deepcopy(clique)
            random.shuffle(random_perm)
            comparator = [
                old_sent == sent for old_sent, sent in zip(old_clique, random_perm)
            ]
            if not np.all(comparator):
                inner_perm.append(random_perm)
                i += 1
            if i >= num_perm:
                break
        permuted_articles.append(inner_perm)
    return permuted_articles


def load_wsj_file_list(data_path):
    dir_list = [
        "00",
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
        "07",
        "08",
        "09",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
        "21",
        "22",
        "23",
        "24",
    ]

    file_list = []
    for dirname in os.listdir(data_path):
        if dirname in dir_list:
            subdirpath = os.path.join(data_path, dirname)
            for filename in os.listdir(subdirpath):
                file_list.append(os.path.join(subdirpath, filename))
    return file_list


def load_file_list(data_name, if_sample):
    if data_name in ["wsj", "wsj_bigram", "wsj_trigram"]:
        if if_sample:
            return load_wsj_file_list(config.SAMPLE_WSJ_DATA_PATH)
        return load_wsj_file_list(config.WSJ_DATA_PATH)
    elif data_name in ["wiki_random", "wiki_bigram_easy"]:
        dir_list = config.WIKI_EASY_TRAIN_LIST + config.WIKI_EASY_TEST_LIST
        if if_sample:
            return load_wiki_file_list(config.SAMPLE_WIKI_DATA_PATH, dir_list)
        return load_wiki_file_list(config.WIKI_EASY_DATA_PATH, dir_list)
    elif (data_name in ["wiki_domain"]) or ("wiki_bigram" in data_name):
        category = data_name[12:]
        if category in config.WIKI_OUT_DOMAIN:
            dir_list = config.WIKI_IN_DOMAIN + [category]
        else:
            dir_list = config.WIKI_IN_DOMAIN
        if if_sample:
            return load_wiki_file_list(config.SAMPLE_WIKI_DATA_PATH, dir_list)
        return load_wiki_file_list(config.WIKI_DATA_PATH, dir_list)
    else:
        raise ValueError("Invalid data name!")


def get_t5(data_name, if_sample=False):
    logging.info("Start parsing...")
    file_list = load_file_list(data_name, if_sample)

    sentences = []
    for file_path in file_list:
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                if (line != "<para_break>") and (line != ""):
                    sentences.append(line)
    logging.info("%d sentences in total." % len(sentences))

    model = T5Model.from_pretrained("t5-small")
    tok = T5Tokenizer.from_pretrained("t5-small")

    enc = tok(
        sentences, return_tensors="pt", truncation=True, padding=True, max_length=512
    )

    # forward pass through encoder only
    output = model.encoder(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        return_dict=True,
    )
    # get the final hidden states
    embeddings = output.last_hidden_state

    embed_dict = dict(zip(sentences, embeddings))
    np.random.seed(0)
    embed_dict["<SOA>"] = np.random.uniform(size=512).astype(np.float32)
    embed_dict["<EOA>"] = np.random.uniform(size=512).astype(np.float32)

    return embed_dict


def get_sbert(data_name, if_sample=False, return_model=False):
    logging.info("Start parsing...")
    file_list = load_file_list(data_name, if_sample)

    sentences = []
    for file_path in file_list:
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                if (line != "<para_break>") and (line != ""):
                    sentences.append(line)
    logging.info("%d sentences in total." % len(sentences))

    logging.info("Loading SBERT models...")
    # params = {
    #     "bsize": 64,
    #     "word_emb_dim": 300,
    #     "enc_lstm_dim": 2048,
    #     "pool_type": "max",
    #     "dpout_model": 0.0,
    #     "version": 1,
    # }
    # model = InferSent(params)
    os.makedirs(config.SBERT_CACHE_PATH, exist_ok=True)
    model = SentenceTransformer(
        "all-mpnet-base-v2", cache_folder=config.SBERT_CACHE_PATH
    )

    # model.load_state_dict(torch.load(config.INFERSENT_MODEL))
    # model.set_w2v_path(config.WORD_EMBEDDING)
    # vocab_size = 10000 if if_sample else 2196017
    # model.build_vocab_k_words(K=vocab_size)
    # if on_gpu:
    #     model.cuda()

    logging.info("Encoding sentences...")
    pool = model.start_multi_process_pool()
    embeddings = model.encode_multi_process(
        sentences,
        pool=pool,
        batch_size=128,
    )
    model.stop_multi_process_pool(pool)
    logging.info("number of sentences encoded: %d" % len(embeddings))

    assert len(sentences) == len(embeddings), "Lengths don't match!"
    embed_dict = dict(zip(sentences, embeddings))
    np.random.seed(0)
    embed_dict["<SOA>"] = np.random.uniform(size=768).astype(np.float32)
    embed_dict["<EOA>"] = np.random.uniform(size=768).astype(np.float32)

    if return_model:
        return embed_dict, model
    else:
        return embed_dict


def get_average_glove(data_name, if_sample=False):
    logging.info("Start parsing...")
    file_list = load_file_list(data_name, if_sample)

    sentences = []
    for file_path in file_list:
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                if (line != "<para_break>") and (line != ""):
                    sentences.append(line)
    logging.info("%d sentences in total." % len(sentences))

    logging.info("Loading glove...")
    word_vec = {}
    with open(config.WORD_EMBEDDING) as f:
        for line in f:
            word, vec = line.split(" ", 1)
            word_vec[word] = np.fromstring(vec, sep=" ")

    embed_dict = {}
    for s in sentences:
        tokens = s.split()
        embed_dict[s] = np.zeros(300, dtype=np.float32)
        sent_len = 0
        for token in tokens:
            if token in word_vec:
                embed_dict[s] += word_vec[token]
                sent_len += 1
        if sent_len > 0:
            embed_dict[s] = np.true_divide(embed_dict[s], sent_len)
    np.random.seed(0)
    embed_dict["<SOA>"] = np.random.uniform(size=300).astype(np.float32)
    embed_dict["<EOA>"] = np.random.uniform(size=300).astype(np.float32)
    return embed_dict


def save_eval_perm(data_name, if_sample=False, random_seed=config.RANDOM_SEED):
    random.seed(random_seed)

    logging.info("Loading valid and test data...")
    if data_name not in config.DATASET:
        raise ValueError("Invalid data name!")
    dataset = DataSet(config.DATASET[data_name])
    # dataset.random_seed = random_seed
    if if_sample:
        valid_dataset = dataset.load_valid_sample()
    else:
        valid_dataset = dataset.load_valid()
    if if_sample:
        test_dataset = dataset.load_test_sample()
    else:
        test_dataset = dataset.load_test()
    valid_df = valid_dataset.article_df
    test_df = test_dataset.article_df

    logging.info("Generating permuted articles...")

    def permute(x):
        x = np.array(x).squeeze()
        # neg_x_list = permute_articles([x], config.NEG_PERM)[0]
        neg_x_list = permute_articles_with_replacement([x], config.NEG_PERM)[0]
        return "<BREAK>".join(["<PUNC>".join(i) for i in neg_x_list])

    valid_df["neg_list"] = valid_df.sentences.map(permute)
    valid_df["sentences"] = valid_df.sentences.map(lambda x: "<PUNC>".join(x))
    valid_nums = valid_df.neg_list.map(lambda x: len(x.split("<BREAK>"))).sum()
    test_df["neg_list"] = test_df.sentences.map(permute)
    test_df["sentences"] = test_df.sentences.map(lambda x: "<PUNC>".join(x))
    test_nums = test_df.neg_list.map(lambda x: len(x.split("<BREAK>"))).sum()

    logging.info("Number of validation pairs %d" % valid_nums)
    logging.info("Number of test pairs %d" % test_nums)

    logging.info("Saving...")
    dataset.save_valid_perm(valid_df)
    dataset.save_test_perm(test_df)
    logging.info("Finished!")


def add_args(parser):
    parser.add_argument("--data_name", type=str, default="wsj_bigram")


if __name__ == "__main__":
    _set_basic_logging()
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    save_eval_perm(args.data_name, False)
