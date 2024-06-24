from pathlib import Path
import os
import glob
import pandas as pd
import chardet 
import nltk
import string
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import cmudict
import spacy
from collections import Counter
import numpy as np

# nlp = spacy.load("en_core_web_sm")
# nlp.max_length = 2000000

def fk_level(text, d):
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.

    Args:
        text (str): The text to analyze.
        d (dict): A dictionary of syllables per word.

    Returns:
        float: The Flesch-Kincaid Grade Level of the text. (higher grade is more difficult)
    """
    word_count = len([w for w in word_tokenize(text) if w not in string.punctuation])
    sentence_count = len(sent_tokenize(text))
    syllables_count = sum(count_syl(w, d) for w in word_tokenize(text) if w not in string.punctuation)
    # score = 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (syllables_count / word_count)
    grade_level = 0.39*(word_count / sentence_count) + 11.8*(syllables_count / word_count) - 15.59
    return grade_level

def syllables(word):
    #referred from stackoverflow.com/questions/14541303/count-the-number-of-syllables-in-a-word
    #This is to estimate the number of syllables in a word based on cluster of vowels
    count = 0
    vowels = 'aeiouy'
    word = word.lower()
    if word[0] in vowels:
        count +=1
    for index in range(1,len(word)):
        if word[index] in vowels and word[index-1] not in vowels:
            count +=1
    if word.endswith('e'):
        count -= 1
    if word.endswith('le'):
        count += 1
    if count == 0:
        count += 1
    return count

def count_syl(word, d):
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters

    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.

    Returns:
        int: The number of syllables in the word.
    """
    try:
        return [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]][0]
    except KeyError:
        #if word not found in cmudict
        return syllables(word)

def detect_encoding(file_path): 
    """Detect the encoding of the file to be read"""
    with open(file_path, 'rb') as file: 
        detector = chardet.universaldetector.UniversalDetector() 
        for line in file: 
            detector.feed(line) 
            if detector.done: 
                break
        detector.close() 
    return detector.result['encoding'] 

def read_novels(path=Path.cwd() / "p1-texts" / "novels"):
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year"""
    all_data = []
    # this for loop will run through folders and subfolders looking for a specific file type
    for root, dirs, files in os.walk(path, topdown=False):
    # look through all the files in the given directory
        for name in files:
            filename = os.path.join(root, name)
            # print(filename)
            name_split = name.replace('_', ' ').replace('.txt','').split('-')
            # print(name_split)
            encoding = detect_encoding(filename)
            with open(filename, 'r', encoding=encoding) as afile:
                # print(filename)
                text = afile.read() # read the file and then add it to the list
                afile.close() # close the file when you're done
            # print(text)
            data = [text, name_split[0], name_split[1], name_split[2]]
            all_data.append(data)
    df = pd.DataFrame(all_data, columns=['Text', 'Title', 'Author', 'Year']).sort_values(by=['Year'], ignore_index=True)
    return df

def parse(df, store_path=Path.cwd() / "pickles", out_name="novels.pkl"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 2000000 
    df['Text_Parsed'] = df['Text'].apply(nlp)
    df.to_pickle(os.path.join(store_path, out_name))
    pass


def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize.
    Additional condition of string punctuation will remove punctuations from the list"""
    tokens = [w for w in word_tokenize(text) if w not in string.punctuation]
    # tokens = nltk.word_tokenize(text)
    ttr = len(set(tokens)) / len(tokens)
    return ttr


def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        results[row["Title"]] = nltk_ttr(row["Text"])
    return results

def get_fks(df, d):
    """helper function to add fk scores to a dataframe"""
    results = {}
    # cmudict = nltk.corpus.cmudict.dict()
    for i, row in df.iterrows():
        results[row["Title"]] = round(fk_level(row["Text"], d), 4)
    return results


def subjects_by_verb_pmi(doc, target_verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    # In essense we need to estimate the co-occurence of subjects wrt the mentioned verb
    # our revised PMI = co-occurence count(subject, verb)/count of subject

    ### Calculate co-occurence of a subject with a targeted verb
    cooccurence_counter = Counter()
    for possible_verb in doc:
        if possible_verb.pos_ == 'VERB' and possible_verb.lemma_ == target_verb:
            for possible_subject in possible_verb.children:
                if possible_subject.dep_ == "nsubj" and not possible_subject.is_stop:
                    cooccurence_counter[possible_subject.text] += 1   
    ### Calculate count of a subject
    subject_count = Counter()
    for possible_subject in doc:
        if possible_subject.dep_ == "nsubj" and not possible_subject.is_stop:
            subject_count[possible_subject.text] += 1
    
    ### calculate pmi of each subject
    pmi_collection = dict()
    for subj, cooccurence_count in cooccurence_counter.items():
        pmi_collection[subj] = cooccurence_count/subject_count[subj]
    
    # pmi_collection_list = [(k, pmi_collection[k]) for k in sorted(pmi_collection, key=pmi_collection.get, reverse=True)]

    return sorted(pmi_collection, key=pmi_collection.get, reverse=True)[0:5]

def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    subject_count = Counter()
    for possible_verb in doc:
        if possible_verb.pos_ == 'VERB' and possible_verb.lemma_ == verb:
            for possible_subject in possible_verb.children:
                if possible_subject.dep_ == "nsubj" and not possible_subject.is_stop:
                    subject_count[possible_subject.text] += 1
    return subject_count.most_common(5)



def subject_counts(doc):
    """Extracts the most common subjects in a parsed document. Returns a list of tuples."""
    subject_count = Counter()
    for possible_subject in doc:
        if possible_subject.dep_ == "nsubj" and not possible_subject.is_stop:
            subject_count[possible_subject.text] += 1
    return subject_count.most_common(5)



if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    # path = Path.cwd() / "p1-texts" / "novels"
    # print(path)
    df = pd.read_pickle(Path.cwd() / "pickles" /"novels.pkl"  )
    # df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    print(df.head())
    # nltk.download("cmudict")
    # parse(df)
    # print(df.head())
    nltk.download('punkt')
    print(get_ttrs(df))
    d = cmudict.dict()
    # print(fk_level(df['Text'][0], d))
    print(get_fks(df, d))
    # df = pd.read_pickle(Path.cwd() / "pickles" /"novels.pkl"  )
    # print(df.head())
    # print(get_subjects(df))

    # For loop to get most common subjects in the doc
    for i, row in df.iterrows():
        print(row["Title"])
        print(subject_counts(row["Text_Parsed"]))
        print("\n")
    for i, row in df.iterrows():
        print(row["Title"])
        print(subjects_by_verb_count(row["Text_Parsed"], "say"))
        print("\n")

    for i, row in df.iterrows():
        print(row["Title"])
        print(subjects_by_verb_pmi(row["Text_Parsed"], "say"))
        print("\n")
