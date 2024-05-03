import nltk
nltk.download('punkt')
sentence = "At␣ eight o'clock on Thursday morning Arthur didn't feel very good."
tokens = nltk.word_tokenize(sentence)
print(tokens)