import chardet
import nltk
import spacy
from collections import Counter
from pathlib import Path



def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    result = chardet.detect(raw_data)
    return result['encoding']


print(Path.cwd())
path = Path.cwd() / "last-week-solution" /"inaugural"
print(path)
#exit()
tokens = []

#1 type token ratio
for file in path.glob("*.txt"):
    year, author = file.stem.split("-")
    text = file.read_text(encoding="utf-8")
    tokens.extend(nltk.word_tokenize(text))

ttr = len(set(tokens)) / len(tokens)

nlp = spacy.load("en_core_web_sm")

#2 counting adjectives
adjectives = Counter()
for file in path.glob("*.txt"):
    year, author = file.stem.split("-")
    text = file.read_text(encoding="utf-8")
    doc = nlp(text)
    for ent in doc.ents:
        print(ent.text, ent.start_char, ent.end_char, ent.label_)
    for token in doc:
        if token.pos_ == "ADJ":
            adjectives[token.lemma_] += 1
print(adjectives.most_common(50))

#3 getting targets of adjectives
targets = Counter()
for file in path.glob("*.txt"):
    year, author = file.stem.split("-")
    text = file.read_text(encoding="utf-8")
    doc = nlp(text)
    for token in doc:
        if token.pos_ == "ADJ":
            for child in token.children:
                if child.pos_ in ("NOUN", "PROPN"):
                    targets[(token.lemma_, child.lemma_)] += 1
print("100 MOST COMMON ADJECTIVE-NOUN PAIRS")
print(targets.most_common(100))
                