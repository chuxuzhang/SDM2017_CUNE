#CUNE-MF: part 2
#Collaborative User Network Embedding for Social Recommender Systems, SDM2017
#Chuxu Zhang (chuxuzhang@gmail.com)

import string
import re
import random
import math
import numpy as np
import sys
import os
from gensim.models import Word2Vec

if len(sys.argv) != 4:
	print "p@1: WalkLen; p@2: DimSize; p@3: WinLen"
	os._exit(0)

WalkLen=string.atoi(sys.argv[1])
DimSize=string.atoi(sys.argv[2])
WinLen=string.atoi(sys.argv[3])

def read_user_user_randWalkSeq(WalkLen):
	walks = []
	inputfile=open("Data/user_randWalkSeq.txt","r")
	for line in inputfile:
		path=[]
		for i in range(WalkLen):
			user_id=re.split(' ',line)[i]			
			path.append(user_id)
		walks.append(path)
	inputfile.close()
	return walks

#read user sequence
user_walks = read_user_user_randWalkSeq(WalkLen)

print("Learn User Embeddings...")
#learn embedding by word2vec model
user_model = Word2Vec(user_walks, size = DimSize, window = WinLen, min_count = 0, workers = 1, sg = 1)
print("Embedding Learnig Finish!")

#save embedding
print("Output User Embeddings...")
user_model.save_word2vec_format("Data/user_embedding.csv")