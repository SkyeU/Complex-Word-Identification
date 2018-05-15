from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from nltk.corpus import brown
import random
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import cess_esp
from nltk.corpus import wordnet as wn
import xgboost as xgb
import numpy as np
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
random.seed(55)
def xg_f1(y,t):
	t = t.get_label()
	y_bin = [1. if y_cont > 0.5 else 0. for y_cont in y] # binaryzing your output
	return 'f1',f1_score(t,y_bin)
class Improved_system(object):

	def __init__(self, language, trainset):
		self.language = language
		pos_tags_nltk = ['cc', 'cd','dt','in','jj','jjr','jjs','nn','nns',    # tag
		'nnp','nnps','pdt','pos','prp','prp$','rb','rbr','rbs','rp',
		'sym','vb','vbd','vbg','vbn','vbp','vbz','wdt','wp','wp$','wrb']
		self.vowels = ['a','e','i','o','u']
		self.vec = CountVectorizer(vocabulary = pos_tags_nltk)  # tag vector

		self.cmudict = nltk.corpus.cmudict.dict()  #syl dict

		self.model = xgb.XGBClassifier(learning_rate =0.1,   # XGboost classifier
						 eta = 1,
						 silent = 1,
						 nround = 10,
						 n_estimators=1000,
						 max_depth=6,
						 min_child_weight=1,
						 gamma=0.1,
						 reg_alpha=0.005,
						 subsample=0.8,
						 colsample_bytree=0.8,
						 objective= 'binary:logistic',
						 nthread=8,
						 scale_pos_weight=1,
						 seed=27)
		self.word_fre = {}
		self.bigram_fre = {}
		self.trigram_fre = {}
		self.tetgram_fre = {}
		for row in trainset:
				sen = row["sentence"]  
				target_word = row["target_word"]
				for word in target_word.split(' '):
					if word in self.word_fre:
						 self.word_fre[word] += 1
					else:
						 self.word_fre[word] = 1 
				for word in sen.split(' '):
					for i in range(len(word) - 1):
						if word[i:i+2] in self.bigram_fre:
							self.bigram_fre[word[i:i+2]] += 1
						else:
							self.bigram_fre[word[i:i+2]] = 0
					for i in range(len(word) - 2):
						if word[i:i+3] in self.trigram_fre:
							self.trigram_fre[word[i:i+3]] += 1
						else:
							self.trigram_fre[word[i:i+3]] = 0

		if self.language == "english":
			self.avg_word_length = 5.3
			brown_corpus =  brown.categories() # brown corpus
			for i in range(len(brown_corpus)):
				file = brown.words(categories=brown_corpus[i])
				for word in file:
					if word not in self.word_fre:
						self.word_fre[word] = 1
					else:
						self.word_fre[word] += 1				
		else: 
			self.avg_word_length = 6.2
			word = cess_esp.words() # spanil corpus
			
			for item in word:
				if item in self.word_fre:
					self.word_fre[item] += 1
				else:
					self.word_fre[item] = 1
			
	def nsyl(self,sen):  # calculate max syllables
		syb_list = []
		for word in sen.split(" "):
			if word.lower() in self.cmudict:
				syb_list.append([len(list(y for y in x if y[-1].isdigit())) for x in self.cmudict[word.lower()]])
		if len(syb_list) > 0:
			return max(syb_list)[0]
		else: 
			return 1                  
						
	def extract_features(self, word, sentence,start_offset,end_offset):
		start_offset = int(start_offset)
		end_offset = int(end_offset)
		word_counts = []
		bi_counts = []
		tri_counts = []
		for token in word.split(" "):
			if token in self.word_fre:
				word_counts.append(self.word_fre[token])   #calculate word count
			else:
				word_counts.append(1)
			for i in range(len(token) - 1):     #calculate bi_character count
				if token[i:i+2] in self.bigram_fre:
					bi_counts.append(self.bigram_fre[token[i:i+2]])
				else:
					bi_counts.append(1)

			for i in range(len(token) - 2):  #calculate tri_character count
				if token[i:i+3] in self.trigram_fre:
					tri_counts.append(self.trigram_fre[token[i:i+3]])
				else:
					tri_counts.append(1)

		if len(word_counts) == 0:   #calculate word frequency
			wd_avg = 1 /len(self.word_fre)
		else: 
			wd_avg = sum(word_counts)/len(word_counts)/len(self.word_fre)

		if len(bi_counts) == 0:   #calculate bi_character frequency
			bi_avg = 1 /len(self.bigram_fre)
		else: 
			bi_avg = sum(bi_counts)/len(bi_counts)/len(self.bigram_fre)

		if len(tri_counts) == 0:   #calculate tri_character frequency
			tri_avg = 1 /len(self.trigram_fre)
		else: 
			tri_avg = sum(tri_counts)/len(tri_counts)/len(self.trigram_fre)
		len_chars = len(word) / self.avg_word_length
		len_tokens = len(word.split(' '))
	   
		num_vowel = 0  #calculate vowels numbers
		num_sense = [len(wn.synsets(token)) for token in word.split(" ") if len(wn.synsets(token)) > 0] # List of senses for each word
	  
		if len(num_sense) > 0:  # calculate max synset number
			avg_sense = sum(num_sense)/len(num_sense)
		else: 
			avg_sense = 0

		for char in word.lower():
			if char in self.vowels:
				num_vowel += 1

		pos = [i[1] for i in nltk.pos_tag(nltk.word_tokenize(sentence))]  #calculate pos_tag vector
		tag_num = len(nltk.word_tokenize(sentence[0:start_offset]))
		pos_word = []
		for i in range(len(word.split(" "))):
			pos_word.append(pos[tag_num+i])
		X = self.vec.fit_transform([' '.join(pos_word)])
		pos_counts = X.toarray()[0]
		if self.language == "english":	
			return[len_chars,len_tokens, avg_sense,num_vowel,wd_avg,self.nsyl(word),bi_avg,tri_avg]+pos_counts.tolist()
		
		else:

			return [len_chars,len_tokens,avg_sense, wd_avg,num_vowel,self.nsyl(word)]#+pos_counts.tolist()
		
					
	def train(self, trainset,devset):
		X = []
		y = []
		X_val = []
		y_val = []
		i = 0
		for sent in trainset:
			# i+=1
			# if i < 1000:
			# print(sent)
				X.append(self.extract_features(sent['target_word'], sent['sentence'],sent['start_offset'],sent['end_offset']))
				y.append(sent['gold_label'])
		for sent in devset:
			# print(sent)
			X_val.append(self.extract_features(sent['target_word'], sent['sentence'],sent['start_offset'],sent['end_offset']))
			y_val.append(sent['gold_label'])

		#self.model.fit(np.array(X),np.array( y),
		#	eval_set=[(np.array(X), np.array( y)), (np.array(X_val), np.array( y_val))],
		#	verbose=False,eval_metric=xg_f1)#,eval_set=(X_val,y_val),verbose=True)

		self.model.fit(np.array(X),np.array( y))
		plot_importance(self.model)
		plt.show()
	def test(self, devset):
		X = []
		for sent in devset:
			X.append(self.extract_features(sent['target_word'], sent['sentence'],sent['start_offset'],sent['end_offset']))

		return self.model.predict(np.array(X))
#        
