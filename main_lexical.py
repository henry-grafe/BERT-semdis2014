import torch
import numpy as np
from transformers import FlaubertModel, FlaubertTokenizer, FlaubertWithLMHeadModel
import json
from tqdm import tqdm
import spacy
import reader_lexical

"""
sentence = "Étalez en couche mince sur du papier absorbant et remuez chaque jour pour favoriser le séchage ."
french_exhaustive_vocabulary = json.load(open("index.json",'rb'))

token_ids = torch.tensor([flaubert_tokenizer.encode(sentence)])
tokens = flaubert_tokenizer.tokenize(sentence)
print("tokens : ", tokens)
print("token ids : ",token_ids)
output = flaubert(token_ids)[0][0][9].detach().to("cpu").numpy()
"""
class SubstitutesExtractor():
    def __init__(self) -> None:
        # Choose among ['flaubert/flaubert_small_cased', 'flaubert/flaubert_base_uncased', 
        #               'flaubert/flaubert_base_cased', 'flaubert/flaubert_large_cased']
        modelname = 'flaubert/flaubert_base_uncased' 

        # Load pretrained model and tokenizer
        self.model, log = FlaubertWithLMHeadModel.from_pretrained(modelname, output_loading_info=True)
        self.tokenizer = FlaubertTokenizer.from_pretrained(modelname, do_lowercase=True)
        self.lemmatizer = spacy.load('fr_core_news_md')
        
        vocab = json.load(open("flaubert_word_vocabulary.json",'r',encoding='utf8'))
        vocab_words = list(vocab.keys())
        vocab_ids = list(vocab.values())
        self.id_to_word_vocab = {}
        for i in range(len(vocab_ids)):
            self.id_to_word_vocab[vocab_ids[i]] = vocab_words[i][:-4]
        self.complete_words_ids = vocab_ids
        self.stop_word = json.load(open("fr_stop_words.json",'r',encoding='utf8'))
        # do_lowercase=False if using cased models, True if using uncased ones
        
    def extract_substitutes(self, context, target_id, top_k = 10):
        split_context = context.split(" ")
        target_word = split_context[target_id]
        target_word_lemma = self.lemmatizer(split_context[target_id])[0].lemma_
        #print(target_word_lemma)
        target_token_id = len(self.tokenizer.encode(" ".join(split_context[:target_id]))) - 1
        #print(self.tokenizer.encode(" ".join(split_context[:target_id])))
        #print(self.tokenizer.encode(context))
        #print(self.tokenizer.tokenize(context))
        #print(target_token_id)
        token_ids = torch.tensor([self.tokenizer.encode(context,max_length=256,
                                                        padding='max_length')])
        token_ids[0][target_token_id] = 4 #replace word by mask
        #print(token_ids)
        #print(token_ids.size())
        #print("passing in the model...")
        output = self.model(token_ids)[0][0][target_token_id].detach().to("cpu").numpy()
        #print("passing done.")
        output_respective_to_ids = output[self.complete_words_ids]
        output_respective_to_ids = np.exp(output_respective_to_ids)/np.exp(output_respective_to_ids).sum()
        index_sorted = np.flip(np.argsort(output_respective_to_ids))
        proposed_words_output = {}
        c_best = 0
        i = 0
        while c_best < top_k:
            current_word_token_id = self.complete_words_ids[index_sorted[i]]
            current_word = self.id_to_word_vocab[current_word_token_id]
            current_score = output_respective_to_ids[index_sorted[i]]
            current_word_lemma = self.lemmatizer(current_word)[0].lemma_
            if (current_word_lemma not in proposed_words_output) and (current_word not in self.stop_word) and (target_word_lemma not in current_word_lemma) and (target_word not in current_word_lemma) and (target_word_lemma not in current_word) and (target_word not in current_word):
                proposed_words_output[current_word_lemma] = current_score
                c_best += 1
            i+=1
        return proposed_words_output
        
extractor = SubstitutesExtractor()
proposed_words = extractor.extract_substitutes("Même si les personnes atteintes d' obésité souffrent des standards esthétiques de la société , l' obésité représente néanmoins une affection aux complications parfois graves . ", 20)
print(proposed_words)

reader = reader_lexical.generate_reader_lexical("dataset/test/lexsubfr_semdis2014_test.xml")

out_f = open("output_file.txt",'w',encoding="utf8")
for main_word in reader:
    instances = reader[main_word]
    for instance in instances:
        instance_id = instance["instance_id"]
        context = instance["clean_context"]
        target_id = instance["target_id"]
        proposed_words = extractor.extract_substitutes(context, target_id, top_k=10)
        print(proposed_words)
        proposed_words = list(proposed_words.keys())
        line = main_word + " " + str(instance_id) + " :: " + proposed_words[0]
        for i in range(1,len(proposed_words)):
            line += " ; " + proposed_words[i]
        line += "\n"
        out_f.write(line)
out_f.close()