# Libraries
import torch
from preprocess import PATH

# Preliminaries
from torchtext.data import Field, TabularDataset, BucketIterator


def get_fields():
    label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
    text_field = Field(tokenize='spacy', lower=True, include_lengths=True, batch_first=True)
    fields = [('label', label_field), ('text', text_field)]
    return text_field, fields


def load_train_iter(device, args):
    # Fields
    text_field, fields = get_fields()

    # Correct paths
    if args.equal:
        train_path = f'train_{args.type}_equal.csv'
        test_path = f'test_{args.type}_equal.csv'
        valid_path = f'valid_{args.type}_equal.csv'
    else:
        train_path = f'train_{args.type}.csv'
        test_path = f'test_{args.type}.csv'
        valid_path = f'valid_{args.type}.csv'

    # TabularDataset
    train, valid, _ = TabularDataset.splits(path=PATH, train=train_path, validation=valid_path, test=test_path,
                                            format='CSV', fields=fields, skip_header=True)

    # Iterators
    train_iter = BucketIterator(train, batch_size=args.batch_size, sort_key=lambda x: len(x.text),
                                device=device, sort=True, sort_within_batch=True)
    valid_iter = BucketIterator(valid, batch_size=args.batch_size, sort_key=lambda x: len(x.text),
                                device=device, sort=True, sort_within_batch=True)

    # Vocabulary
    text_field.build_vocab(train, min_freq=3)

    return text_field, train_iter, valid_iter


def load_test_iter(device, args, path):
    text_field, fields = get_fields()

    # TabularDataset
    test = TabularDataset(path=path, format='CSV', fields=fields, skip_header=True)

    # Iterator
    test_iter = BucketIterator(test, batch_size=args.batch_size, sort_key=lambda x: len(x.text),
                               device=device, sort=True, sort_within_batch=True)

    text_field.build_vocab(test, min_freq=3)

    return test_iter
