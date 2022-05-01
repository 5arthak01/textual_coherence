import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import os


class WSJ_Bigram_Dataset(Dataset):
    def __init__(self, scr_path, portion=1.0, mode="train"):
        self.data_path = scr_path
        self.train_list = [
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
        ]
        self.valid_list = ["11", "12", "13"]
        self.test_list = [
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
        self.portion = portion
        self.mode = mode

        if self.mode == "train":
            self.file_list = self.get_file_list(self.train_list)
        elif self.mode == "valid":
            self.file_list = self.get_file_list(self.valid_list)
        elif self.mode == "test":
            self.file_list = self.get_file_list(self.test_list)
        else:
            raise ValueError("Invalid mode name!")

        self.sentences = self.get_all_sentences(self.file_list)
        self.total_sent = len(self.sentences)
        self.examples = []

        self.sent_index, self.article_index = self.create_index(self.file_list)
        self.sent_df = pd.DataFrame(self.sent_index)
        self.sent_df.columns = ["article", "sentences"]
        self.sent_df.set_index("article", inplace=True)
        self.article_df = pd.DataFrame(self.article_index)
        self.article_df.columns = ["article", "sentences"]
        self.article_df.reset_index(level=0, inplace=True)

        if self.mode in ["train"]:
            self.total_cliques = len(self.examples)
        else:
            self.total_cliques = len(self.article_df)

    def __len__(self):
        return self.total_cliques

    def __getitem__(self, index):
        if self.mode in ["train"]:
            article = self.examples[index]
            samples = self.sent_df.loc[article, "sentences"]
            sample = np.random.choice(samples)
            return sample
        else:
            article_row = self.article_df.loc[index]
            return article_row["article"]

    def get_file_list(self, dir_list):
        file_list = []
        for dirname in os.listdir(self.data_path):
            if dirname in dir_list:
                subdirpath = os.path.join(self.data_path, dirname)
                for filename in os.listdir(subdirpath):
                    file_list.append(os.path.join(subdirpath, filename))
        return file_list

    def preprocess(self, article, index):
        sentences = []
        with open(article) as f:
            for line in f:
                line = line.strip()
                if line != "<para_break>" and line != "":
                    sentences.append(line)
        sent_num = len(sentences)
        if sent_num < 3:
            return sentences
        sentences = ["<SOA>"] + sentences + ["<EOA>"]
        sent1 = sentences[:-1]
        pos_sent2 = sentences[1:]
        samples = []
        weights = []
        for i in range(sent_num + 1):
            for j in list(range(i)) + list(range(i + 2, sent_num + 2)):
                sent = "<BREAK>".join(
                    [sent1[i], pos_sent2[i], sentences[j], str(sent_num)]
                )
                samples.append(sent)
                factor = np.sqrt(max(1, np.abs(i - j)))
                weights.append(1.0 / factor)
        index.append(
            [
                article,
                np.random.choice(
                    samples, max(1, int(len(samples) * self.portion)), False
                ),
            ]
        )
        for _ in range(50):
            self.examples.append(article)
        return sentences[1:-1]

    def create_index(self, file_list):
        sidx = []
        aidx = []
        for article in file_list:
            sentences = self.preprocess(article, sidx)
            if len(sentences) > 2:
                aidx.append([article, sentences])
        return sidx, aidx

    def get_all_sentences(self, file_list):
        sentences = []
        for article in file_list:
            with open(article) as f:
                for line in f:
                    line = line.strip()
                    if line != "<para_break>" and line != "":
                        sentences.append(line)
        return sentences


class DataSet:
    def __init__(self, d):
        self.dataset = d["dataset"]
        self.data_path = d["data_path"]
        self.sample_path = d["sample_path"]
        self.valid_perm = d["valid_perm"]
        self.test_perm = d["test_perm"]
        self.kwargs = d["kwargs"]
        self.col_names = ["article", "sentences", "neg_list"]

    def load_train(self, portion=1.0):
        return self.dataset(self.data_path, portion, "train", **self.kwargs)

    def load_valid(self, portion=1.0):
        return self.dataset(self.data_path, portion, "valid", **self.kwargs)

    def load_test(self, portion=1.0, **kwargs):
        kwargs.update(self.kwargs)
        return self.dataset(self.data_path, portion, "test", **kwargs)

    def load_train_sample(self):
        return self.dataset(self.sample_path, 1.0, "train", **self.kwargs)

    def load_valid_sample(self):
        return self.dataset(self.sample_path, 1.0, "valid", **self.kwargs)

    def load_test_sample(self):
        return self.dataset(self.sample_path, 1.0, "test", **self.kwargs)

    def load_valid_perm(self):
        df = pd.read_csv(
            self.valid_perm, sep="\t", names=["article", "sentences", "neg_list"]
        )
        df.set_index("article", inplace=True)
        return df

    def load_test_perm(self):
        df = pd.read_csv(
            self.test_perm, sep="\t", names=["article", "sentences", "neg_list"]
        )
        df.set_index("article", inplace=True)
        return df

    def save_valid_perm(self, df):
        df[self.col_names].to_csv(self.valid_perm, sep="\t", index=False, header=False)

    def save_test_perm(self, df):
        df[self.col_names].to_csv(self.test_perm, sep="\t", index=False, header=False)
