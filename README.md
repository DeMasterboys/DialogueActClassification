# Influence of Emotion Peturbations on Dialogue Act Classification

## Downloading dataset
1. Download the DialyDialog dataset from: https://www.aclweb.org/anthology/I17-1099/
2. Extract the EMNLP_dataset folder and put it in the GitHub folder
3. In the EMNLP_dataset folder also extract train.zip, train.zip and validation.zip

## Running the model
### Training
For training the model with standard parameters use:
```python
  python train.py
```

The training parameters can be changed using the command line and are defined as:
```python
  parser.add_argument('--batch_size', default=32, type=int)
  parser.add_argument('--epochs', default=10, type=int)
  parser.add_argument('--lr', default=0.0001, type=float)
  parser.add_argument('--type', default='act', type=str, choices={'emotion', 'act'})
  parser.add_argument('--test', default=False, type=bool)
  parser.add_argument('--equal', default=False, type=bool)
  parser.add_argument('--length', default=np.inf, type=float)
```

To use an equally distributed dataset run the following command:
```python
  python train.py --equal True
```

### Testing
For testing the model when trained normally use:
```python
  python train.py --test True
```

For testing the model when trained on a equally distributed dataset use:
```python
  python train.py --test True --equal True
```
