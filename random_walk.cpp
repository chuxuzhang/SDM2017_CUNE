//CUNE-MF: part 1
//Collaborative User Network Embedding for Social Recommender Systems, SDM2017
//Chuxu Zhang (chuxuzhang@gmail.com)

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <random>
#include <algorithm>
#include <vector>
using namespace std;

void rand_walk_seq_gen(int, int, int, int, char*);

int main(int argc, char *argv[]) 
{
	if(argc!=6)
	{
		cout<<"p@1: UserMaxId; p@2: ItemMaxId; p@3: FilePath; p@4: WalkNum; p@5: WalkLen"<<endl;
		return 0;
	}

	//set parameters of random walk sequence
	int U_maxId = atoi(argv[1]);
	int I_maxId = atoi(argv[2]);
	int WalkNum = atoi(argv[4]);
	int WalkLen = atoi(argv[5]);

	//train/test data split + generate user sequences by random walk
	srand((unsigned)time(NULL));
	rand_walk_seq_gen(U_maxId, I_maxId, WalkNum, WalkLen, argv[3]);

	return 0;
}

void rand_walk_seq_gen(int U_maxId, int I_maxId, int WalkNum, int WalkLen, char* File)
{
	cout<<"train/test data split ..."<<endl;
	//user_item preference neighbor lists in train data
	vector<vector<int> > train_I_neigh_U;
	train_I_neigh_U.resize(I_maxId+1);
	//vector<vector<int> > train_U_neigh_I;
	//train_U_neigh_I.resize(U_maxId+1);

	//train/test data split and user_item preference neighbor generation 
	int userId, itemId;
	float rating;
	int count = 0;
	FILE *rating_file = fopen(File,"r");
	FILE *rating_train_file = fopen("Data/ratings_train.csv","w");
	FILE *rating_test_file = fopen("Data/ratings_test.csv","w");
	while(!feof(rating_file))
	{	
		count++;
		fscanf(rating_file, "%d\t%d\t%f\n", &userId, &itemId, &rating);
		//random split of train/test data
		if(count%10 == 1 || count%10 == 2 || count%10 == 3 || count%10 == 5 || count%10 == 6 || count%10 == 7|| count%10 == 8 || count%10 == 9)
		{
			//rating > thredshold, set it as positive preference feedback
			if(rating >= 3)
			{
				//train_U_neigh_I[userId].push_back(itemId);
				train_I_neigh_U[itemId].push_back(userId);
			}
			fprintf(rating_train_file, "%d %d %f\n", userId, itemId, rating);
		}
		else
		{
			fprintf(rating_test_file, "%d %d %f\n", userId, itemId, rating);	
		}
	}
	fclose(rating_file);
	fclose(rating_train_file);
	fclose(rating_test_file);
	cout<<"train/test data split finish!"<<endl;

	//generate user_user projection network 
	//users who have common items in preference neighbor list are connected in projection network
	vector<vector<int> > train_U_neigh_U;
	train_U_neigh_U.resize(U_maxId+1);
	vector<int>::iterator I1, I2;
	for(int I_j = 1; I_j <= I_maxId; I_j++)
	{
		for(I1 = train_I_neigh_U[I_j].begin(); I1 != train_I_neigh_U[I_j].end(); I1++)
			for(I2 = train_I_neigh_U[I_j].begin(); I2 != train_I_neigh_U[I_j].end(); I2++)
				if(*I1 != *I2)
					train_U_neigh_U[*I1].push_back(*I2);
	}

	cout<<"generate user sequences ..."<<endl;
	//generate user sequences by random walk on projection network
	//according to experiment, random walk has close performance to bias random walk
	FILE *randWalkSeq=fopen("Data/user_randWalkSeq.txt","w");
	int currentNode;
	for(int u_i = 1; u_i <= U_maxId; u_i++)
	{
		if(train_U_neigh_U[u_i].size())
		{
			for(int j = 1; j <= WalkNum; j++)
			{
				currentNode = u_i;
				fprintf(randWalkSeq,"%d ",currentNode);
				for(int l = 1; l < WalkLen; l++)
				{
					currentNode=train_U_neigh_U[currentNode][rand()%train_U_neigh_U[currentNode].size()];
					fprintf(randWalkSeq,"%d ",currentNode);
				}
				fprintf(randWalkSeq,"\n");
			}
		}
	}
	fclose(randWalkSeq);
	cout<<"generate user sequences finish!"<<endl;
}

