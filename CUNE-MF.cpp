//CUNE-MF: part 3
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

vector<int> U_Embed_Label;
vector<int> U_Train_Label;
vector<int> I_Train_Label;
vector<vector<int> > U_TopK_U;
vector<vector<float> > U_Embed;
vector<vector<float> > U_LatentF;
vector<vector<float> > I_LatentF;

void pars_init(int, int, int, int);
void gen_top_k_neigh(int, int, int, int);
void CUNE_MF(int, float, float, float);

const int max_iter=1e4;
double global_ave=0;

int main(int argc, char *argv[]) 
{
	if(argc!=9)
	{
		cout<<"p@1: UserMaxId; p@2: ItemMaxId; p@3: LearnRate; p@4: TopK; p@5: EmbedSize; p@6: LatentSize; p@7: LamReg; p@8: SocLamReg"<<endl;
		return 0;
	}

	//set pars
	int U_maxId = atoi(argv[1]);
	int I_maxId = atoi(argv[2]);
	float LearnRate = atof(argv[3]);
	int TopK = atoi(argv[4]);
	int EmbedS= atoi(argv[5]);
	int LatentS= atoi(argv[6]);
	float LamReg = atof(argv[7]);
	float SocLamReg = atof(argv[8]);

	srand((unsigned)time(NULL));
	pars_init(U_maxId, I_maxId, EmbedS, LatentS);//pars init
	gen_top_k_neigh(U_maxId, I_maxId, EmbedS, TopK);//top-k neighbors generation

	CUNE_MF(LatentS, LearnRate, LamReg, SocLamReg);
}

void pars_init(int U_maxId, int I_maxId, int EmbedS, int LatentS)
{
	U_Embed_Label.resize(U_maxId+1);
	U_Train_Label.resize(U_maxId+1);
	I_Train_Label.resize(I_maxId+1);
	U_Embed.resize(U_maxId+1);
	U_LatentF.resize(U_maxId+1);
	I_LatentF.resize(I_maxId+1);

	for(int U_i = 1; U_i <= U_maxId; U_i++)
	{
		U_Train_Label[U_i]=0;
		U_Embed_Label[U_i]=0;
		U_Embed[U_i].resize(EmbedS+1);
		U_LatentF[U_i].resize(LatentS+1);

		for(int k = 1; k <= EmbedS; k++)
			U_Embed[U_i][k]=(float)rand() / (RAND_MAX);//random init of embeddings

		for(int kk = 1; kk <= LatentS; kk++)
			U_LatentF[U_i][kk]=(float)rand() / (RAND_MAX);//random init of user latent features
	}

	for(int I_j = 1; I_j <= I_maxId; I_j++)
	{
		I_Train_Label[I_j]=0;
		I_LatentF[I_j].resize(LatentS+1);
		for(int k = 1; k <= LatentS; k++)
			I_LatentF[I_j][k]=(float)rand() / (RAND_MAX);//random init of item latent features
	}

	int train_sample_size=0;
	int userId, itemId;
	double rating, rating_sum=0;
	FILE *train_sample_file=fopen("Data/ratings_train.csv","r");
	while(!feof(train_sample_file))
	{
		fscanf(train_sample_file,"%d %d %lf\n",&userId,&itemId,&rating);
		U_Train_Label[userId]=1;
		I_Train_Label[itemId]=1;
		train_sample_size++;
		rating_sum+=rating;
	}
	fclose(train_sample_file);
	global_ave=rating_sum/train_sample_size;

}

bool pair_descend_order(pair<float, int> a, pair<float, int> b)        
{        
    return a.first > b.first;        
}  

void gen_top_k_neigh(int U_maxId, int I_maxId, int EmbedS, int TopK)
{
	//read user embeddings 
	cout<<"read user embeddings ..."<<endl;
	ifstream UserEmbedFile("Data/user_embedding.csv");
	char line[10000]={0};
	int index, embed_index, U_id;
	while(UserEmbedFile.getline(line, sizeof(line)))
	{
		char *tokenPtr=strtok(line," "); 
		index = 0;
		embed_index=0;
		while(tokenPtr != NULL)
		{
			index++;
			if(index == 1)
			{
				U_id=atoi(tokenPtr);
				U_Embed_Label[U_id]=1;
			}
			else
			{
				embed_index++;
				U_Embed[U_id][embed_index]=atof(tokenPtr);
			}
			tokenPtr=strtok(NULL," ");
		}
	}
	UserEmbedFile.close();
	cout<<"read user embeddings finish!"<<endl;

	//generate TopK neighbors of users
	cout<<"generate TopK neighbors ..."<<endl;
	U_TopK_U.resize(U_maxId+1);
	float scoreT;
	for(int U_i = 1; U_i <= U_maxId; U_i++)
	{
		if(U_Embed_Label[U_i])
		{
			vector<pair<float, int> > ScoreList; 
			for(int U_i_2 = 1; U_i_2 <= U_maxId; U_i_2++)
			{
				if(U_i_2!=U_i)
				{
					scoreT=0;
					for(int k = 1; k <= EmbedS; k++)
						scoreT += U_Embed[U_i][k] * U_Embed[U_i_2][k];
					ScoreList.push_back(make_pair<float, int>(scoreT, U_i_2)); 
				}
			}
			sort(ScoreList.begin(), ScoreList.end(), pair_descend_order); 

			for(int j = 0; j < TopK; j++)
				U_TopK_U[U_i].push_back(ScoreList[j].second);
		}
	}
	cout<<"generate TopK neighbors finish!"<<endl;
}

void CUNE_MF_Evaluate(int LatentS)
{
	int userId, itemId;
	double rating;
	double scoreT, MAE_sum=0, RMSE_sum=0;
	int test_sample_size=0;
	FILE *test_sample_file=fopen("Data/ratings_test.csv","r");
	while(!feof(test_sample_file))
	{
		fscanf(test_sample_file,"%d %d %lf\n",&userId,&itemId,&rating);
		// test_sample_size++;
		// if(U_Train_Label[userId] == 0 || I_Train_Label[itemId] == 0)
		// {
		// 	MAE_sum+=abs(rating - global_ave);
		// 	RMSE_sum+=(rating - global_ave) * (rating - global_ave);
		// }
		// else
		if(U_Train_Label[userId] && I_Train_Label[itemId])
		{
			test_sample_size++;
			scoreT=0;
			for(int k = 1; k <= LatentS; k++)
			scoreT+=U_LatentF[userId][k]*I_LatentF[itemId][k];

			MAE_sum+=abs(rating - scoreT);
			RMSE_sum+=(rating - scoreT) * (rating - scoreT);
		}	
	}
	fclose(test_sample_file);

	float MAE_ave = MAE_sum / test_sample_size;
	float RMSE_ave = sqrt(RMSE_sum / test_sample_size);
	cout<<MAE_ave<<"  "<<RMSE_ave<<endl;

}

void CUNE_MF(int LatentS, float LearnRate, float LamReg, float SocLamReg)
{
	cout<<"CUNE_MF training ..."<<endl;
	cout<<"iteration  "<<"MAE  "<<"RMSE  "<<endl;
	int userId, itemId;
	double rating;
	double scoreT, score_diff;
	vector<int>::iterator I;
	for(int t = 0; t < max_iter; t++)
	{
		FILE *train_sample_file=fopen("Data/ratings_train.csv","r");
		while(!feof(train_sample_file))
		{
			fscanf(train_sample_file,"%d %d %lf\n",&userId,&itemId,&rating);
			//cout<<rating<<endl;
			//U_Train_Label[userId]=1;
			//I_Train_Label[itemId]=1;

			scoreT=0;
			for(int k = 1; k <= LatentS; k++)
			scoreT+=U_LatentF[userId][k]*I_LatentF[itemId][k];

			score_diff=rating - scoreT;

			for(int kk = 1; kk <= LatentS; kk++)
			{
				U_LatentF[userId][kk] -= LearnRate * (-score_diff * I_LatentF[itemId][kk] + LamReg * U_LatentF[userId][kk]);
				I_LatentF[itemId][kk] -= LearnRate * (-score_diff * U_LatentF[userId][kk] + LamReg * I_LatentF[itemId][kk]);
			}

			if(U_Embed_Label[userId])
			{
				float neigh_diff_sum=0;
				for(int l = 1; l <= LatentS; l++)
				{
					neigh_diff_sum=0;
					for(I = U_TopK_U[userId].begin(); I != U_TopK_U[userId].end(); I++)
					neigh_diff_sum += (U_LatentF[userId][l] - U_LatentF[*I][l]);

					U_LatentF[userId][l] -= LearnRate * SocLamReg * neigh_diff_sum;
				}
			}
		}
		fclose(train_sample_file);

		if(t % 10 ==0)
		{
			cout<<t<<"  ";
			CUNE_MF_Evaluate(LatentS);	
		}	
	}
}


