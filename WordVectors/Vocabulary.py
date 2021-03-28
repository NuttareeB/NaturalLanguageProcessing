from collections import Counter 
from re import sub, compile
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
import string

class UnimplementedFunctionError(Exception):
	pass

class Vocabulary:

	def __init__(self, corpus):

		self.word2idx, self.idx2word, self.freq = self.build_vocab(corpus)
		self.size = len(self.word2idx)

	def most_common(self, k):
		freq = sorted(self.freq.items(), key=lambda x: x[1], reverse=True)
		return [t for t,f in freq[:k]]


	def text2idx(self, text):
		tokens = self.tokenize(text)
		return [self.word2idx[t] if t in self.word2idx.keys() else self.word2idx['UNK'] for t in tokens]

	def idx2text(self, idxs):
		return [self.idx2word[i] if i in self.idx2word.keys() else 'UNK' for i in idxs]


	def getpostag(self, t):
		wordnet_tag = t[0].lower()
		return wordnet_tag if wordnet_tag in ['a', 'r', 'n', 'v'] else 'n'

	###########################
	## TASK 1.1           	 ##
	###########################
	def tokenize(self, text):
		"""
	    
	    tokenize takes in a string of text and returns an array of strings splitting the text into discrete tokens.

	    :params: 
	    - text: a string to be tokenize, e.g. "The blue dog jumped, but not high."

	    :returns:
	    - tokens: a list of strings derived from the text, e.g. ["the", "blue", "dog", "jumped", "but", "not", "high"] for word-level tokenization
	    
	    """ 
		lemmatizer = WordNetLemmatizer()
		newtext = text.translate(str.maketrans('', '', string.punctuation)).lower()

		text_list = newtext.split()
		
		text_w_postag = nltk.pos_tag(text_list)

		# # lemmatize with POS-tag
		# tokens = [lemmatizer.lemmatize(word, self.getpostag(tag)) for word, tag in text_w_postag]

		# lemmatize without POS-tag
		tokens = [lemmatizer.lemmatize(word) for word, tag in text_w_postag]

		# print("len(tokens):", len(tokens))
		# print("lemmatize_stem_text:", tokens)

		return tokens



	###########################
	## TASK 1.2            	 ##
	###########################
	def build_vocab(self,corpus):
		"""
	    
	    build_vocab takes in list of strings corresponding to a text corpus, tokenizes the strings, and builds a finite vocabulary

	    :params:
	    - corpus: a list string to build a vocabulary over

	    :returns: 
	    - word2idx: a dictionary mapping token strings to their numerical index in the dictionary e.g. { "dog": 0, "but":1, ..., "UNK":129}
	    - idx2word: the inverse of word2idx mapping an index in the vocabulary to its word e.g. {0: "dog", 1:"but", ..., 129:"UNK"}
	    - freq: a dictionary of words and frequency counts over the corpus (including words not in the dictionary), e.g. {"dog":102, "the": 18023, ...}

	    """ 
		tokens = []
		for c in corpus:
			tokens += self.tokenize(c)
		word_freq = Counter(tokens)
		i = 0
		high_feq_word = {}
		freq = {}
		for w, count in word_freq.most_common():
			freq[w] = count
			if count >= 50:
				high_feq_word[w] = i
				i += 1
		high_feq_word["UNK"] = i
		word2idx = high_feq_word
		# print(high_feq_word)

		idx2word = {i: w for w, i in word2idx.items()}

		# print(idx2word)

		# print(freq)
		return word2idx, idx2word, freq


	###########################
	## TASK 1.3              ##
	###########################
	def make_vocab_charts(self):
		"""
	    
	    make_vocab_charts plots word frequency and cumulative coverage charts for this vocabulary. See handout for more details

	    
	    """ 

		#Token Frequency Distribution graph
		plt.title("Token Frequency Distribution")
		plt.xlabel("Token ID (sorted by frequency)")
		plt.ylabel("Frequency")
		plt.plot(self.freq.values())
		plt.plot(np.full(len(self.freq.values()), 50), 'r')
		plt.text(80000, 55, "freq=50", color="red")
	#     plt.legend(["train", "dev"], loc ="lower left") 
		plt.yscale('log')
		plt.show(block=False)
		# plt.close()
		# plt.pause(0.01)

		#Cumulative Fraction Covered graph
		plt.title("Cumulative Fraction Covered")
		plt.xlabel("Token ID (sorted by frequency)")
		plt.ylabel("Fraction of Token Occurences Covered")

		freq_list = list(self.freq.values())

		x_axis = 0
		for i in range(len(freq_list)):
			if freq_list[i] < 50:
				x_axis = i
				break
		
		print("x_axis:",  x_axis)

		cumulative_freq_list = []
		for i in range(len(freq_list)):
			if i == 0:
				cumulative_freq_list.append(freq_list[i])
			else:
				cumulative_freq_list.append(freq_list[i] + cumulative_freq_list[i-1])
		sum_count = sum(freq_list)
		cumulative_freq_list = [c/sum_count for c in cumulative_freq_list]
		plt.plot(cumulative_freq_list)
		plt.plot(np.full(2, x_axis), range(2), 'r')
		plt.text(x_axis+100, cumulative_freq_list[x_axis]-0.02, cumulative_freq_list[x_axis], color="red")
		plt.show(block=False)

	    # # REMOVE THIS ONCE YOU IMPLEMENT THIS FUNCTION
		# raise UnimplementedFunctionError("You have not yet implemented make_vocab_charts.")

