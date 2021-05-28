import numpy as np
from preprocess import preprocess_text


def has_illegal_char(string):
    # Removes unwanted words and numbers
    ignore = ['.', ',', '-', '?', '$', ';', '(', ')']
    return any(char.isdigit() or char in ignore for char in string)


def percentage_same_number(items):
    # Finds the emotion which occures most often with the percentage that it occurs
    unique_items = set(items)
    dic = {x:0 for x in unique_items}
    total = len(items)
    for numb in items:
        dic[numb] += 1
    max_value = max(dic.values())
    max_keys = [(k,v/total*100) for k, v in dic.items() if v == max_value]
    return max_keys


def get_emotion_act_frequency():
    # Load the data
    df = preprocess_text('train', 'both')
    # text = df['text'].to_numpy()
    # print(text)

    # Puts all words in a dict with the act and emotion for each sentence containing the word
    dic = {}
    for act, emotion, text in df.to_numpy():
        # Emotion can't be no emotion (0)
        if emotion == 0:
            continue

        # Stemming
        # sno = SnowballStemmer('english')
        # sno.stem(word)

        # Look at unique and wanted words in a sentence
        sentence = set(word.lower() for word in text.strip().split() if not has_illegal_char(word) and not word == '\'')
        for word in sentence:
            if word in dic:
                # Add act and emotion to list of word in dict
                acts, emotions = dic[word]
                acts.append(act), emotions.append(emotion)
            else:
                # Add new word to dict
                dic[word] = ([act], [emotion])
    return dic


def find_relevant_keys(dic, print_stats=True):
    relevant_key_list = []
    for key, value in dic.items():
        max_keys = percentage_same_number(value[1])
        # Shows words that are mainly used for 1 emotion (excluding happiness) which occurs at least 6 times
        if len(value[1]) >= 6 and max_keys[0][0] != 4 and max_keys[0][1] > 60:
            relevant_statistics = [key, max_keys[0][0], np.round(max_keys[0][1], 2), len(value[1])]
            relevant_key_list.append(relevant_statistics)
            if print_stats:
                #print(' {} & {} &  {} &  {} \\\ '.format(*relevant_statistics))
                print('Key {}:\t\t Dominant emotion {}, percentage {}, number of appearance {}'
                            .format(key, max_keys[0][0], max_keys[0][1], len(value[1])))
                # print(value[1])
        elif len(value[1]) >= 100 and max_keys[0][0] == 4 and max_keys[0][1] > 90:
            relevant_statistics = [key, max_keys[0][0], np.round(max_keys[0][1], 2), len(value[1])]
            relevant_key_list.append(relevant_statistics)
            if print_stats:
                #print(' {} & {} &  {} &  {} \\\ '.format(*relevant_statistics))
                print('Key {}:\t\t Dominant emotion {}, percentage {}, number of appearance {}'
                            .format(key, max_keys[0][0], max_keys[0][1], len(value[1])))
                # print(value[1])
    return np.array(relevant_key_list)


def main():
    dic = get_emotion_act_frequency()
    _ = find_relevant_keys(dic)


if __name__ == '__main__':
    main()
