import string
import re
from unicodedata import normalize
from pickle import dump,load
from collections import Counter

def load_doc(filename):
    file = open(filename)
    text = file.read()
    file.close()
    return text

def get_sentence(doc):
    return doc.strip().split('\n')


#English data
# filename1 = 'English.txt'
# english = load_doc(filename1)
# english_sentences = get_sentence(english)
# print("English corpus: %d" %(len(english_sentences)))

# #Dutch data
# filename2 = 'Dutch.txt'
# dutch = load_doc(filename2)
# dutch_sentences = get_sentence(dutch)
# print("Dutch corpus: %d" %(len(dutch_sentences)))

def cleaned_corpus(lines):
    clean_corpus =  []
    #filter printable chars
    re_pat = re.compile('[^%s]' % re.escape(string.printable))
    #making a transtable to remove punctuation chars
    table = str.maketrans('','',string.punctuation)
    for line in lines:
        line = normalize('NFD',line).encode('ascii','ignore') #Normalizing chars such as o` and o:
        line = line.decode('UTF-8')
        line = line.split() #tokenize on whitespaces
        line = [word.lower() for word in line] #lowercase all the words
        line = [word.translate(table) for word in line] #remove all the punctuation marks
        line = [re_pat.sub('',word) for word in line] #remove non-printable chars
        line = [word for word in line if word.isalpha()] #remove numbers
        clean_corpus.append(' '.join(line))
    return clean_corpus

# english_cleaned = clean_corpus(english_sentences)
# dump(english_cleaned,open('English_cleaned.pkl','wb'))
# dutch_cleaned = cleaned_corpus(dutch_sentences)
# dump(dutch_cleaned,open('Dutch_cleaned.pkl','wb'))

def vocab_cnt(sentences):
    voc = Counter()
    for l in sentences:
        words = l.split()
        voc.update(words)
    return voc



# d = (vocab_cnt(lines_eng))
# c = 0
# for i in d:
#     print(i)
#     if c == 10:
#         break
#     c += 1

# print('English vocab',(vocab_cnt(lines_eng)))
# print('Dutch vocab',(vocab_cnt(lines_dutch)))


#Removing words
def remove_voc(voc,threshold):
    saved = [word for word,cnt in voc.items() if cnt >= threshold]
    return (set(saved))


#Updating the data set
def update_dataset(sentences, vocab):
    new_sentences = []
    for line in sentences:
        new_words = []
        for t in line.split():
            if t in vocab:
                new_words.append(t)
            else:
                new_words.append('uup')
        new_words = [word for word in new_words if word !='s']
        new_line = ' '.join(new_words)
        new_sentences.append(new_line)
    return new_sentences

filename3 = 'English_cleaned.pkl'
lines_eng = load(open(filename3, 'rb'))
filename4 = 'Dutch_cleaned.pkl'
lines_dutch = load(open(filename4,'rb'))

vocab_eng = vocab_cnt(lines_eng)
new_vocab_eng = remove_voc(vocab_eng,4)
print('Old vocab size eng: %d New vocab size eng: %d'%(len(vocab_eng),len(new_vocab_eng)))
new_data_eng = update_dataset(lines_eng,new_vocab_eng)
for i in range(10):
    print(new_data_eng[i])
dump(new_data_eng,open('English_cleaned_voc.pkl','wb'))
print("Saved english!")

vocab_dutch = vocab_cnt(lines_dutch)
new_vocab_dutch = remove_voc(vocab_dutch,4)
print('Old vocab size dutch: %d New vocab size dutch %d'%(len(vocab_dutch),len(new_vocab_dutch)))
new_data_dutch = update_dataset(lines_dutch,new_vocab_dutch)
dump(new_data_dutch,open('Dutch_cleaned_voc.pkl','wb'))
print("Saved dutch!")
