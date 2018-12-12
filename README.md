code of CUNE-MF for SDM2017 paper: Collaborative User Network Embedding for Social Recommender Systems
Author: Chuxu Zhang (chuxuzhang@gmail.com)

<1> step-1: run randomwalk.cpp to generate train/test split dataset and user sequences

#parameters 
p@1: UserMaxId (maximum user id); p@2: ItemMaxId (maximum item id); p@3: FilePath (input data); p@4: WalkNum (# of walk rooted at each node); p@5: WalkLen (random walk length)

#output
ratings_train.csv: training data;  ratings_test.csv: test data; user_randWalkSeq.txt: user sequences file

#example on CiaoDVD dataset

g++ -o random_walk random_walk.cpp

./random_walk 17615 16121 Data/ratings_new.csv 20 20

<2> step-2: run net_embed.py to generate user embeddings (need gensim api: https://radimrehurek.com/gensim/)

#parameters 
p@1: WalkLen (random walk length); p@2: DimSize (dimension of embedding); p@3: WinLen (window size)

#output 
user_embedding.csv: user embeddings file

#example on CiaoDVD dataset

python net_embed.py 20 20 5

<3> step-3: run CUNE-MF.cpp to generate top-k neighbors, train model as well as evaluate model's performance 

#parameters 
p@1: UserMaxId (maximum user id); p@2: ItemMaxId (maximum item id); p@3: LearnRate (learning rate); p@4: TopK (number of top-k neighbors); p@5: EmbedSize (dimension of embeddings); p@6: LatentSize (dimension of latent features); p@7: LamReg (regularization parameter); p@8: SocLamReg (regularization parameter of top-k neighbors)

#output
evalution results: MAE and RMSE

#example on CiaoDVD dataset

g++ -o CUNE-MF CUNE-MF.cpp

./CUNE-MF 17615 16121 0.001 50 20 20 0.1 0.1
