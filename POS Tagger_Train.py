from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
import winsound
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


def untag(tagged_sentence):
    return [w for w, t in tagged_sentence]


def transform_to_dataset(tagged_sentences):
    X, y = [], []

    for tagged in tagged_sentences:
        for index in range(len(tagged)):
            X.append(features(untag(tagged), index))
            y.append(tagged[index][1])

    return X, y


def pos_tag(sentence):
    tags = clf.predict([features(sentence, index) for index in range(len(sentence))])
    return list(zip(sentence, tags))


tokens = []
with open("Merged_Bhojpuri_dataset", "r", encoding="utf-8-sig") as sentences_file:
    reader = sentences_file
    for row in reader:
        tokens.append(row.split())

tagged_sentences = []
temp = 0
for x in range(0, len(tokens)):
    if not tokens[x]:
        sentence = tokens[temp:x]
        sentence = [tuple(x) for x in sentence if len(x) >= 2]
        temp = x+1
        tagged_sentences.append(sentence)

print("Total sentences: ", len(tagged_sentences))

cutoff = int(.75 * len(tagged_sentences))
training_sentences = tagged_sentences[:cutoff]
test_sentences = tagged_sentences[cutoff:]

print("Total Training Sentences: ", len(training_sentences))
print("Total Testing Sentences: ", len(test_sentences))

X, y = transform_to_dataset(training_sentences)

clf = Pipeline([
    ('vectorizer', DictVectorizer(sparse=False)),
    ('classifier', DecisionTreeClassifier(criterion='entropy'))
])

clf.fit(X[:len(training_sentences)], y[:len(training_sentences)])

print('Training completed')
winsound.Beep(900, 400)

X_test, y_test = transform_to_dataset(test_sentences)

print("Accuracy:", clf.score(X_test, y_test))

joblib.dump(clf, 'model.pkl')
