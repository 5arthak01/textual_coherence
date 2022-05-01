O/P for `run_bigram_coherence`
After "Training BigramCoherence model..."

- fit() `coherence_models.py` : 143
- evaluate_dis() `coherence_models.py` : 238 prints all sentences

`results_discrimination.zip` has results of hyperparameter tuning for `run_bigram_coherence` (without sigmoid)

- `baseline.json` is the baseline results for `run_bigram_coherence`

# Analysis of Hyperparameter tuning with discrimination validation

# Comparison of baseline results

| BASELINE       | Discrimination      | Insertion           |
| -------------- | ------------------- | ------------------- |
| AVG_GLOVE      | 0.9253795066413663  | 0.2984713791414052  |
| SIGMOID        | 0.8039373814041746  | 0.2146964623152086  |
| TANH AVG_GLOVE | 0.10488614800759014 | 0.7206061213667566  |
| SBERT          | 0.9385199240986717  | 0.33034726781926527 |

### Hyparameters

- input_dropout: [0.5, 0.6, 0.7]
- hidden_layers: [1, 2]
- hidden_dropout: [0.2, 0.3, 0.4]
- margin: [4.0, 5.0, 6.0]
- weight_decay: [0.0, 0.1]
- dpout_model: [0.0, 0.05, 0.1]

### Results for best 20

##### Scores

- Discrimination

  | Description | Value   |
  | ----------- | ------- |
  | mean        | 0.92649 |
  | std         | 0.00107 |
  | min         | 0.925   |
  | max         | 0.9291  |
  | 25%         | 0.92585 |
  | 50%         | 0.92635 |
  | 75%         | 0.9269  |

- Insertion

  | Description | Value    |
  | ----------- | -------- |
  | mean        | 0.30562  |
  | std         | 0.00156  |
  | min         | 0.3036   |
  | max         | 0.3091   |
  | 25%         | 0.304275 |
  | 50%         | 0.3054   |
  | 75%         | 0.306825 |

##### Trends

- input_dropout: 0.5 > 0.6
- hidden_layers: 2 > 1
- margin: 6.0 > 4.0 > 5.0
- weight_decay: 0.0
- dpout_model: 0.0 > 0.1 > 0.05
- hidden_dropout
  - Discrimination: 0.3 > 0.2 > 0.4
  - Insertion: 0.3 > 0.4 > 0.2

### Best results

- Same model performs best in both tasks

```python
{
    "hparams": {
        "loss": "margin",
        "input_dropout": 0.5,
        "hidden_state": 500,
        "hidden_layers": 2,
        "hidden_dropout": 0.3,
        "num_epochs": 50,
        "margin": 6.0,
        "lr": 0.001,
        "weight_decay": 0.0,
        "use_bn": False,
        "task": "discrimination",
        "bidirectional": False,
        "dpout_model": 0.0
    },
}
```

# Analysis of Hyperparameter tuning with insertion validation

### Results for best 20

##### Scores

- Discrimination

  | Description | Value    |
  | ----------- | -------- |
  | mean        | 0.926410 |
  | std         | 0.001753 |
  | min         | 0.923900 |
  | max         | 0.930300 |
  | 25%         | 0.925100 |
  | 50%         | 0.925900 |
  | 75%         | 0.927425 |

- Insertion

  | Description | Value  |
  | ----------- | ------ |
  | mean        | 0.3041 |
  | std         | 0.0029 |
  | min         | 0.301  |
  | max         | 0.3124 |
  | 25%         | 0.3024 |
  | 50%         | 0.3033 |
  | 75%         | 0.305  |

##### Trends

- input_dropout: 0.5
- hidden_layers: 1 > 2
- hidden_dropout: 0.4 > 0.2 > 0.3
- margin: 6.0 > 5.0 > 4.0
- weight_decay: 0.0
- dpout_model: 0.05 > 0.1 > 0.0

### Best results

- Same model performs best in both tasks

```python
{
    "hparams": {
        "loss": "margin",
        "input_dropout": 0.5,
        "hidden_state": 500,
        "hidden_layers": 1,
        "hidden_dropout": 0.4,
        "num_epochs": 50,
        "margin": 6.0,
        "lr": 0.001,
        "l2_reg_lambda": 0.0,
        "use_bn": False,
        "task": "insertion",
        "bidirectional": False,
        "dpout_model": 0.05
    }
}
```

# Results

### After Hyperparameter tuning

| Sr. No. | Validation Task | Bidirectional | Output function | Encoder       |
| ------- | --------------- | ------------- | --------------- | ------------- |
| 0       | discrimination  | False         | None            | average_glove |
| 1       | discrimination  | False         | None            | sbert         |
| 2       | discrimination  | False         | tanh            | average_glove |
| 3       | discrimination  | False         | tanh            | sbert         |
| 4       | discrimination  | False         | sigmoid         | average_glove |
| 5       | discrimination  | False         | sigmoid         | sbert         |
| 6       | discrimination  | True          | None            | average_glove |
| 7       | discrimination  | True          | None            | sbert         |
| 8       | discrimination  | True          | tanh            | average_glove |
| 9       | discrimination  | True          | tanh            | sbert         |
| 10      | discrimination  | True          | sigmoid         | average_glove |
| 11      | discrimination  | True          | sigmoid         | sbert         |
| 12      | insertion       | False         | None            | average_glove |
| 13      | insertion       | False         | None            | sbert         |
| 14      | insertion       | False         | tanh            | average_glove |
| 15      | insertion       | False         | tanh            | sbert         |
| 16      | insertion       | False         | sigmoid         | average_glove |
| 17      | insertion       | False         | sigmoid         | sbert         |
| 18      | insertion       | True          | None            | average_glove |
| 19      | insertion       | True          | None            | sbert         |
| 20      | insertion       | True          | tanh            | average_glove |
| 21      | insertion       | True          | tanh            | sbert         |
| 22      | insertion       | True          | sigmoid         | average_glove |
| 23      | insertion       | True          | sigmoid         | sbert         |
