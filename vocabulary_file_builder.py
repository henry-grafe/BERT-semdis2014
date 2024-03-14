import json
from tqdm import tqdm
from transformers import FlaubertModel, FlaubertTokenizer, FlaubertWithLMHeadModel

modelname = 'flaubert/flaubert_base_uncased' 

flaubert_tokenizer = FlaubertTokenizer.from_pretrained(modelname, do_lowercase=False)
vocabulary = flaubert_tokenizer.get_vocab()
new_voc = {}
for k,v in vocabulary.items():
    new_voc[v] = k
for i in range(100000):
    if ("<" in new_voc[i]) and ("</w>" not in new_voc[i]):
        print(i, new_voc[i])
input('next')
french_exhaustive_vocabulary = json.load(open("index.json",'rb'))

word_vocabulary = {}
tokens = list(vocabulary.keys())
token_ids = list(vocabulary.values())
for i in tqdm(range(len(vocabulary))):
    if len(tokens[i]) > 4:
        if tokens[i][-4:] == "</w>":
            if tokens[i][:-4] in french_exhaustive_vocabulary:
                word_vocabulary[tokens[i]] = token_ids[i]


json.dump(word_vocabulary, open("flaubert_word_vocabulary.json",'w'))
print("Length of flaubert vocabulary : ",len(word_vocabulary))

