
from datasets import load_dataset
from Vocabulary import Vocabulary
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.utils.extmath import randomized_svd
import logging
import itertools
import math
from sklearn.manifold import TSNE
import os.path

import random
random.seed(42)
np.random.seed(42)

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

class UnimplementedFunctionError(Exception):
	pass


###########################
## TASK 2.2              ##
###########################

def compute_cooccurrence_matrix(corpus, vocab):
	"""
	    
	    compute_cooccurrence_matrix takes in list of strings corresponding to a text corpus and a vocabulary of size N and returns 
	    an N x N count matrix as described in the handout. It is up to the student to define the context of a word

	    :params:
	    - corpus: a list strings corresponding to a text corpus
	    - vocab: a Vocabulary object derived from the corpus with N words

	    :returns: 
	    - C: a N x N matrix where the i,j'th entry is the co-occurrence frequency from the corpus between token i and j in the vocabulary

	    """ 
	
	if os.path.isfile('occo.npy') and os.path.isfile('no_contexts.npy'):
		loaded_data = np.load('occo.npy')
		loaded_context_size = np.load('no_contexts.npy')
		return loaded_data, loaded_context_size

	cooccurrence_matrix = np.zeros((vocab.size, vocab.size))
	context_size = 5

	no_contexts = 0
	for sentense in corpus:
		# tokens = vocab.tokenize(sentense)
		idx_list = vocab.text2idx(sentense)
		no_contexts += len(idx_list) - (2*context_size)
		for i, idx in enumerate(idx_list):
			if i < context_size or i > len(idx_list) - context_size - 1:
				continue
			cooccurrence_matrix[idx][idx] += 1
			counted_word = set([idx])
			# print(countexd_word)
			for j in range(i-context_size, i + context_size):
				j_position = idx_list[j]
				if j_position not in counted_word:
					cooccurrence_matrix[idx][j_position] += 1
					counted_word.add(j_position)
	np.save('occo.npy', cooccurrence_matrix)
	np.save('no_contexts.npy', no_contexts)

	return cooccurrence_matrix, no_contexts
	

###########################
## TASK 2.3              ##
###########################

def compute_ppmi_matrix(corpus, vocab):
	# print("vocab:", vocab)
	"""
	    
	    compute_ppmi_matrix takes in list of strings corresponding to a text corpus and a vocabulary of size N and returns 
	    an N x N positive pointwise mutual information matrix as described in the handout. Use the compute_cooccurrence_matrix function. 

	    :params:
	    - corpus: a list strings corresponding to a text corpus
	    - vocab: a Vocabulary object derived from the corpus with N words

	    :returns: 
	    - PPMI: a N x N matrix where the i,j'th entry is the estimated PPMI from the corpus between token i and j in the vocabulary

	    """ 
	cm, no_contexts = compute_cooccurrence_matrix(corpus, vocab)
	cm = cm + 0.000001
	print(cm)

	if os.path.isfile('ppmi.npy'):
		loaded_data = np.load('ppmi.npy')

		return loaded_data

	ppmi = np.zeros((vocab.size, vocab.size))

	for i in range(len(cm)):
		for j in range(len(cm)):
			ppmi[i][j] = max(0, (np.log((cm[i][j]*no_contexts)/(cm[i][i]*cm[j][j]))))
			if math.isnan(ppmi[i][j]):
				ppmi[i][j] = 0
			
	np.save('ppmi.npy', ppmi)

	print(ppmi)
	return ppmi


	

################################################################################################
# Main Skeleton Code Driver
################################################################################################
def main_freq():

	logging.info("Loading dataset")
	dataset = load_dataset("ag_news")
	dataset_text =  [r['text'] for r in dataset['train']]
	dataset_labels = [r['label'] for r in dataset['train']]


	logging.info("Building vocabulary")
	vocab = Vocabulary(dataset_text)
	vocab.make_vocab_charts()
	plt.close()
	plt.pause(0.01)


	logging.info("Computing PPMI matrix")
	PPMI = compute_ppmi_matrix( [doc['text'] for doc in dataset['train']], vocab)


	logging.info("Performing Truncated SVD to reduce dimensionality")
	word_vectors = dim_reduce(PPMI)


	logging.info("Preparing T-SNE plot")
	plot_word_vectors_tsne(word_vectors, vocab)


def dim_reduce(PPMI, k=16):
	U, Sigma, VT = randomized_svd(PPMI, n_components=k, n_iter=10, random_state=42)
	SqrtSigma = np.sqrt(Sigma)[np.newaxis,:]

	U = U*SqrtSigma
	V = VT.T*SqrtSigma

	word_vectors = np.concatenate( (U, V), axis=1) 
	word_vectors = word_vectors / np.linalg.norm(word_vectors, axis=1)[:,np.newaxis]

	return word_vectors


def plot_word_vectors_tsne(word_vectors, vocab):
	coords = TSNE(metric="cosine", perplexity=50, random_state=42).fit_transform(word_vectors)

	plt.cla()
	top_word_idx = vocab.text2idx(" ".join(vocab.most_common(1000)))
	plt.plot(coords[top_word_idx,0], coords[top_word_idx,1], 'o', markerfacecolor='none', markeredgecolor='k', alpha=0.5, markersize=3)

	for i in tqdm(top_word_idx):
		plt.annotate(vocab.idx2text([i])[0],
			xy=(coords[i,0],coords[i,1]),
			xytext=(5, 2),
			textcoords='offset points',
			ha='right',
			va='bottom',
			fontsize=5)
	plt.show()


if __name__ == "__main__":
    main_freq()

