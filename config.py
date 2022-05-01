from utils.data_utils import WSJ_Bigram_Dataset

# ------------------- PATH -------------------
ROOT_PATH = "."
DATA_PATH = "%s/data" % ROOT_PATH
LOG_PATH = "%s/log" % ROOT_PATH
CHECKPOINT_PATH = "%s/checkpoint" % ROOT_PATH
RESULTS_PATH = "%s/results" % ROOT_PATH

# ------------------- DATA -------------------

SBERT_CACHE_PATH = "%s/sbert_cache" % DATA_PATH
WORD_EMBEDDING = "%s/glove.840B.300d.txt" % DATA_PATH

DATASET = {}

WSJ_DATA_PATH = "%s/parsed_wsj" % DATA_PATH
SAMPLE_WSJ_DATA_PATH = "%s/parsed_wsj" % DATA_PATH
WSJ_VALID_PERM = "%s/valid_perm.tsv" % WSJ_DATA_PATH
WSJ_TEST_PERM = "%s/test_perm.tsv" % WSJ_DATA_PATH

DATASET["wsj_bigram"] = {
    "dataset": WSJ_Bigram_Dataset,
    "data_path": WSJ_DATA_PATH,
    "sample_path": SAMPLE_WSJ_DATA_PATH,
    "valid_perm": WSJ_VALID_PERM,
    "test_perm": WSJ_TEST_PERM,
    "kwargs": {},
}


# ------------------- PARAM ------------------

RANDOM_SEED = 2018

NEG_PERM = 20
