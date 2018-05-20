import os
import re
from nltk import tokenize
#from nltk.tokenize import sent_tokenize, word_tokenize

#from gtts.tokenizer import pre_processors, Tokenizer, tokenizer_cases
#from gtts.utils import _minimize, _len, _clean_tokens
#from gtts.lang import tts_langs


text = "Otherwise...       program will--    split the?! dataset that you! feed to it -   in the: training mode itself. In the directory with; the training data you will  ? find three data files with the following extensions"
print(text)
punct_marks = ['.', ',', ':', ';', '!', '?', '--', '-', '...', '?!', '??', '!!', '!?']

words = tokenize.word_tokenize(text)
print(words)
simple_sentences = [[]]
for w in words:
    if w in punct_marks:
        if len(simple_sentences[-1]) != 0:
            simple_sentences[-1].append(w)
            simple_sentences.append([])
    else:
        simple_sentences[-1].append(w.lower())
if len(simple_sentences[-1]) == 0:
    simple_sentences.pop()
if not simple_sentences[-1] in punct_marks:
    simple_sentences[-1].append('.')

print(simple_sentences)
#tokens = tokenize.
#print(tokens)

czloid_path = 'D:\Projects\SpeechSynthesis\CZloid VCCV 2015'
folders = list()
for folder in os.listdir(czloid_path):
    if os.path.isdir(os.path.join(czloid_path, folder)):
        folders.append(folder)

lyric_list = set()
for folder in folders:
    folder_path = os.path.join(czloid_path, folder)
    with open(os.path.join(folder_path, 'oto.ini'), 'r') as oto_file:
        while True:
            line = oto_file.readline()
            if not line:
                break
            p1 = line.find('=')
            if p1 == -1:
                continue
            file_name = line[0:p1-4]
            lyric_params = line.split(',')
            if lyric_params[0] == '':
                lyric_params[0] = file_name
            #lyric_list.add(lyric)

#print(lyricSet)
