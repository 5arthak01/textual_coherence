# Dataset

- [[Link]](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/sarthak_agrawal_research_iiit_ac_in/Ea3Se4CFeVdMnATyIwQiufcB8tUgsj2Bj29FStIUiOjPaw?e=wLMfwK) to download the dataset.
- Name of dataset is `wsj_bigram`
- Save in `./data/` directory relactive to `run_model.py` file.

# Preprocessing

Premute the original documents or paragraphs to obtain the negative samples for evaluation:

```
python preprocess.py
```

### Training and Evaluation

To train and evaluate a model, run

```
python run_model.py --sent_encoder <sent_encoder>
```

where `sent_encoder` can be "average_glove" or "sbert" (sbert by default).

# Checkpoints

[[Link]](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/sarthak_agrawal_research_iiit_ac_in/EYPh9CDmi3NLsJzd7MvXPBEBzT18RiZE015cRdQLYfDSrg?e=WobcjH) for checkpoints of the trained baseline models.

# Code Structure

- Main model is in `models/coherence_models.py`
- The NN used in the model is at `models/discriminator_models.py`
