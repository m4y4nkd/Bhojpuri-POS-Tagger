from nltk import word_tokenize
from sklearn.externals import joblib


def features(sentence, index):
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
    }


def pos_tag(sentence):
    clf = joblib.load('model.pkl')
    tags = clf.predict([features(sentence, index) for index in range(len(sentence))])
    return list(zip(sentence, tags))


string = input("Enter the string to POS Tag: ")
print(pos_tag(word_tokenize(string)))
