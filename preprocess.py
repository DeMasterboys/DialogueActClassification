import pandas as pd
import os
from collections import Counter
import random

DATASET = "EMNLP_dataset"
PATH = 'csv'


def get_act(file):
    # Loads the acts from file into a list
    if file == 'all':
        path = f"{DATASET}/dialogues_act.txt"
    else:
        path = f"{DATASET}/{file}/dialogues_act_{file}.txt"

    with open(path, encoding="utf8") as act_file:
        act = act_file.read()
        act = act.replace('\n', '').replace(' ', '')
        split_act = [int(x) for x in act]
    return split_act


def get_emotion(file):
    # Loads the emotions from file into a list
    if file == 'all':
        path = f"{DATASET}/dialogues_emotion.txt"
    else:
        path = f"{DATASET}/{file}/dialogues_emotion_{file}.txt"

    with open(path, encoding="utf8") as emotion_file:
        emotion = emotion_file.read()
        emotion = emotion.replace('\n', '').replace(' ', '')
        split_emotion = [int(x) for x in emotion]
    return split_emotion


def get_text(file):
    # Loads the text from file into a list
    if file == 'all':
        path = f"{DATASET}/dialogues_text.txt"
    else:
        path = f"{DATASET}/{file}/dialogues_{file}.txt"

    with open(path, encoding="utf8") as sentence_file:
        sentence_list = []
        for i, line in enumerate(sentence_file):
            split_line = line.split('__eou__')
            del split_line[-1]  # newline character
            sentence_list += split_line
        if file == 'all':
            del sentence_list[-1]
    return sentence_list


def preprocess_text(file, class_type, length, test=False):
    # Puts the emotion/act and text from the different text files into a dataframe
    if class_type == 'act':
        split_act = get_act(file)
        df = pd.DataFrame(split_act, columns=['act'])
        df['act'] = df['act'] - 1

    elif class_type == 'emotion':
        split_emotion = get_emotion(file)
        df = pd.DataFrame(split_emotion, columns=['emotion'])

    elif class_type == 'both':
        split_act = get_act(file)
        df = pd.DataFrame(split_act, columns=['act'])
        df['act'] = df['act'] - 1

        split_emotion = get_emotion(file)
        df['emotion'] = split_emotion

    sentence_list = get_text(file)
    df['text'] = sentence_list
    if test:
        df = df[df['text'].apply(lambda line: len(line.split()) < length)]
    return df


def make_df_equal(df, all_data=False):
    # Find act with lowest amount of occurences
    act_list = df['act'].to_list()
    if not act_list:
        print("Dataframe is empty")
        return pd.DataFrame([])
    count = Counter(act_list)
    lowest_amount = count.most_common(4)[-1][1]
    new_act_list = []
    new_emotion_list = []
    new_text_list = []

    # For every act slice same amount of occurences
    for unique_act in set(act_list):
        # find all data for current act
        temp_df = df[df['act'] == unique_act]
        temp_act = temp_df['act'].to_list()
        temp_text = temp_df['text'].to_list()
        if all_data:
            # shuffle act, text and emotion
            temp_emotion = temp_df['emotion'].to_list()
            shuffled_lists = list(zip(temp_act, temp_emotion, temp_text))
            random.shuffle(shuffled_lists)
            shuffled_temp_act, shuffled_temp_emotion, shuffled_temp_text = zip(*shuffled_lists)
            new_emotion_list.extend(shuffled_temp_emotion[:lowest_amount])
        else:
            # shuffle act and text
            shuffled_lists = list(zip(temp_act, temp_text))
            random.shuffle(shuffled_lists)
            shuffled_temp_act, shuffled_temp_text = zip(*shuffled_lists)

        # add sliced amount to total list
        new_act_list.extend(shuffled_temp_act[:lowest_amount])
        new_text_list.extend(shuffled_temp_text[:lowest_amount])

    # create new equally distributed dataframe
    equal_df = pd.DataFrame(new_act_list, columns=['act'])
    equal_df['text'] = new_text_list
    if all_data:
        equal_df['emotion'] = new_emotion_list
    return equal_df


def get_csv_files(args):
    try:
        os.mkdir(PATH)
    except OSError:
        pass

    # Preprocess text
    df_train = preprocess_text('train', args.type, args.length)
    df_validation = preprocess_text('validation', args.type, args.length)
    df_test = preprocess_text('test', args.type, args.length, test=True)

    if args.equal:
        # make equal
        df_train_equal = make_df_equal(df_train)
        df_validation_equal = make_df_equal(df_validation)
        df_test_equal = make_df_equal(df_test)

        # save csv files
        df_train_equal.to_csv(f'{PATH}/train_{args.type}_equal.csv', index=False)
        df_validation_equal.to_csv(f'{PATH}/valid_{args.type}_equal.csv', index=False)
        df_test_equal.to_csv(f'{PATH}/test_{args.type}_equal.csv', index=False)
    else:
        # Save csv files
        df_train.to_csv(f'{PATH}/train_{args.type}.csv', index=False)
        df_validation.to_csv(f'{PATH}/valid_{args.type}.csv', index=False)
        df_test.to_csv(f'{PATH}/test_{args.type}.csv', index=False)
