import os
from nltk.tokenize import sent_tokenize, word_tokenize

text = "Otherwise,       program will    split the dataset that you feed to it    in the training mode itself. In the directory with the training data you will find three data files with the following extensions."
print(text)

sentences = sent_tokenize(text)
tokens = list()
for s in sentences:
    tokens.append(word_tokenize(s))
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
            lyric_list.append(lyric)

print(lyricSet)
