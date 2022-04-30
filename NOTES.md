O/P for `run_bigram_coherence`
After "Training BigramCoherence model..."

- fit() `coherence_models.py` : 143
- evaluate_dis() `coherence_models.py` : 238 prints all sentences

`results_discrimination.zip` has results of hyperparameter tuning for `run_bigram_coherence` (without sigmoid)

- `baseline.json` is the baseline results for `run_bigram_coherence`

# Analysis of Hyperparameter tuning with discrimination validation

### Hyparameters

- input_dropout: [0.5, 0.6, 0.7]
- hidden_layers: [1, 2]
- hidden_dropout: [0.2, 0.30000000000000004, 0.4000000000000001]
- margin: [4.0, 5.0, 6.0]
- weight_decay: [0.0, 0.1]
- dpout_model: [0.0, 0.05, 0.1]

### Results for best 20

##### Scores

['0.9291', '0.9286', '0.9279', '0.9274', '0.9269', '0.9269', '0.9265', '0.9265', '0.9264', '0.9264', '0.9263', '0.9263', '0.9259', '0.9259', '0.9259', '0.9257', '0.9256', '0.9254', '0.9252', '0.9250']

['0.3091', '0.3082', '0.3072', '0.3069', '0.3069', '0.3068', '0.3064', '0.3062', '0.3059', '0.3054', '0.3054', '0.3048', '0.3048', '0.3046', '0.3043', '0.3042', '0.3041', '0.3040', '0.3036', '0.3036']

##### Discrimination

- input_dropout: 0.5 > 0.6
- hidden_layers: 2 > 1
- hidden_dropout: 0.30000000000000004 > 0.2 > 0.4000000000000001
- margin: 6.0 > 4.0 > 5.0
- weight_decay: 0.0
- dpout_model: 0.0 > 0.1 > 0.05

##### Insertion

- input_dropout: 0.5 > 0.6
- hidden_layers: 2 > 1
- hidden_dropout: 0.30000000000000004 > 0.4000000000000001 > 0.2
- margin: 6.0 > 4.0 > 5.0
- weight_decay: 0.0
- dpout_model: 0.0 > 0.1 > 0.05

### Best results:

```python
print(trimmed_disc_models[0])
{'hparams': {'loss': 'margin', 'input_dropout': 0.5, 'hidden_state': 500, 'hidden_layers': 2, 'hidden_dropout': 0.30000000000000004, 'num_epochs': 50, 'margin': 6.0, 'lr': 0.001, 'weight_decay': 0.0, 'use_bn': False, 'task': 'discrimination', 'bidirectional': False, 'dpout_model': 0.0}, 'discrimination': [0.9291271347248576, [0.8881799163179916, 0.9248603351955307, 0.980352644836272]], 'insertion': [0.30914630954998895, [0.49057581191472405, 0.23112018911437762, 0.12588024765485295]]}

print(trimmed_ins_models[0])
{'hparams': {'loss': 'margin', 'input_dropout': 0.5, 'hidden_state': 500, 'hidden_layers': 2, 'hidden_dropout': 0.30000000000000004, 'num_epochs': 50, 'margin': 6.0, 'lr': 0.001, 'weight_decay': 0.0, 'use_bn': False, 'task': 'discrimination', 'bidirectional': False, 'dpout_model': 0.0}, 'discrimination': [0.9291271347248576, [0.8881799163179916, 0.9248603351955307, 0.980352644836272]], 'insertion': [0.30914630954998895, [0.49057581191472405, 0.23112018911437762, 0.12588024765485295]]}
```
