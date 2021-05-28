import pandas as pd
import random

from preprocess import PATH, preprocess_text, make_df_equal

key_list = {0: 'no emotion', 1: 'anger', 2: 'disgust', 3: 'fear', 4: 'happiness', 5: 'sadness', 6: 'surprise'}
emotion_categories = [0, 1, 2, 3, 4, 5, 6]
words_per_act = {
    1: ['damn', 'stupid', 'shut', 'annoying', 'rude', 'complaint'],
    2: ['gross', 'dirty'],
    3: ['scared', 'afraid'],
    4: ['good', 'great', 'fun', 'love', 'interesting', 'wonderful', 'nice', 'perfect', 'beautiful', 'pleasure', 'happy', 'enjoy', 'glad', 'appreciate'],
    5: ['pity', 'sorry', 'apologize', 'ill', 'awfully'],
    6: ['surprising', 'kidding', 'surprise', 'incredible', 'unbelievable', 'goodness', 'surprised', 'huh', 'oops']
}
strong_words_dict = {
    'damn': 1, 'stupid': 1, 'shut': 1, 'annoying': 1, 'rude': 1, 'complaint': 1,
    'gross': 2, 'dirty': 2,
    'scared': 3, 'afraid': 3,
    'good': 4, 'great': 4, 'fun': 4, 'love': 4, 'interesting': 4, 'wonderful': 4, 'nice': 4, 'perfect': 4, 'beautiful': 4, 'pleasure': 4, 'happy': 4, 'enjoy': 4, 'glad': 4, 'appreciate': 4,
    'pity': 5, 'sorry': 5, 'apologize': 5, 'ill': 5, 'awfully': 5,
    'surprising': 6, 'kidding': 6, 'surprise': 6, 'incredible': 6, 'unbelievable': 6, 'goodness': 6, 'surprised': 6, 'huh': 6, 'oops': 6
}
strong_words = ['damn', 'stupid', 'shut', 'annoying', 'rude', 'complaint',
                'gross', 'dirty',
                'scared', 'afraid',
                'good', 'great', 'fun', 'love', 'interesting', 'wonderful', 'nice', 'perfect', 'beautiful', 'pleasure', 'happy', 'enjoy', 'glad', 'appreciate',
                'pity', 'sorry', 'apologize', 'ill', 'awfully',
                'surprising', 'kidding', 'surprise', 'incredible', 'unbelievable', 'goodness', 'surprised', 'huh', 'oops']


def intersection(lst1, lst2):
    # finds intersection between two lists
    return list(set(lst1) & set(lst2))


def get_test_set(length=10, print_change=False):
    # find most relevant keys and dominant emotions
    # dic = get_emotion_act_frequency()
    # relevant_keys = find_relevant_keys(dic, print_stats=False)
    # keys = list(relevant_keys[:, 0])
    # emotions = [int(x) for x in relevant_keys[:, 1]]

    # construct df
    df = preprocess_text('test', 'both', length)

    # create new lists to build dataset
    act_list = []
    emotion_list = []
    text_list = []

    for (act, emotion, text) in df.to_numpy():
        if len(text.split()) <= length:
            # add to dataset if length matches
            act_list.append(act)
            emotion_list.append(emotion)
            text_list.append(text)

            text = text.lower()
            words = intersection(strong_words, text.split())
            # if current sentence contains dominant word
            if len(words) > 0:
                if len(words) == 1:
                    word = words[0]
                else:
                    count = 0
                    word = ''
                    for possible_word in words:
                        if strong_words_dict[possible_word] == act:
                            word = possible_word
                            count += 1
                    # only not choose random when count == 1
                    if count != 1:
                        word = random.choice(words)

                # find current emotion and create list with new possible emotions
                word_emotion = strong_words_dict[word]
                emotion_categories_copy = emotion_categories[:]
                emotion_categories_copy.remove(word_emotion)

                # add every other emotion to dataset
                for emotion_choice in emotion_categories_copy:
                    # pass if new dominant emotion is not present in analysis
                    if emotion_choice not in words_per_act.keys():
                        continue

                    # add all words from new emotion to dataset
                    for new_word in words_per_act[emotion_choice]:
                        new_text = text.replace(word, new_word)

                        # add old_act, new_emotion and new_text to dataset
                        act_list.append(act)
                        emotion_list.append(emotion_choice)
                        text_list.append(new_text)

                        if print_change:
                            print(f"old text: {text}")
                            print(f"{word} ({key_list[word_emotion]}) -> {new_word} ({key_list[emotion_choice]})")
                            print(f"new text: {new_text}")
                            print()

    # Create new dataframe
    new_df = pd.DataFrame(act_list, columns=['act'])
    new_df['emotion'] = emotion_list
    new_df['text'] = text_list
    new_df.to_csv(f'{PATH}/testset.csv', index=False)

    # Create dataframe with equal distribution
    equal_df = make_df_equal(new_df, all_data=True)
    equal_df.to_csv(f'{PATH}/testset_equal.csv', index=False)


def main():
    get_test_set()


if __name__ == '__main__':
    main()
