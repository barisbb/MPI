
/*
Student Name: Barış Büyüktaş
Notes  : To run the code         1) mpic++ -o cmpe300_mpi_2020800003 ./cmpe300_mpi_2020800003.cpp
                                 2) mpirun --oversubscribe -np num_of_processor cmpe300_mpi_2020800003 /...folder_path.../mpi_project_dev0.tsv
/**/

#include <mpi.h>
using namespace std;
#include <fstream>
#include <string>
#include <vector>
#include <iterator>
#include <sstream>
#include <iostream>
#include <limits>
#include <typeinfo>
#include <algorithm>
#include <cmath>
vector<string> strings;
vector<vector<string> > items;
int numOfProcessor;

int instances;
int features;
int iterations;
int top_features;
struct timespec delta = {0 /*secs*/, 100000000 /*nanosecs*/};
void read_tsv(string fname) {
	ifstream folder(fname);
	string line;
	getline(folder, line);
	numOfProcessor=stod(line);
	getline(folder, line);


	istringstream values(line);
	for(string s; values >> line; )
		strings.push_back(line);
	instances=stod(strings[0]);
	features=stod(strings[1]);
	iterations=stod(strings[2]);

	top_features=stod(strings[3]);
	while (getline(folder, line)) {
		stringstream value(line);
		vector<string> item;
		string tmp;
		while(getline(value, tmp, '\t')) {

			item.push_back(tmp);
		}

		items.push_back(item);
	}


}
vector<int> manhattan(vector<vector<float> > data,int order){ //finds the index of hit and miss, order specifies the index of target instance

	vector<int> hit_miss; //includes 2 integers, nearest hit and nearest miss
	vector<int> same; //indexes of the instances, which have the same labels with the target instance
	vector<int> different;//indexes of the instances, which have the different labels with the target instance

	int iter_label=data[order][data[0].size()-1];


	int index_miss;
	int index_hit;
	for(int i=0;i<data.size();i++){

		if(i != order){

			if((int)data[i][data[0].size()-1]==iter_label){

				same.push_back(i);

			}
			else{
				different.push_back(i);
			}
		}
	}

	int min_hit=1000000;
	for(int i=0;i<same.size();i++){

		int sum=0;
		for(int j=0;j<data[0].size()-1;j++){
			sum = sum + abs(data[order][j]-data[same[i]][j]);
		}

		if(sum<min_hit){
			index_hit=same[i];
			min_hit=sum;
		}
	}

	hit_miss.push_back(index_hit);
	int min_miss=1000000;
	for(int i=0;i<different.size();i++){

		int sum=0;
		for(int j=0;j<data[0].size()-1;j++){
			sum = sum + abs(data[order][j]-data[different[i]][j]);
		}

		if(sum<min_miss){
			index_miss=different[i];
			min_miss=sum;
		}
	}

	hit_miss.push_back(index_miss);
	return hit_miss;

}
float difference_low_high(vector<vector<float> > items,int order){ //finds the difference of maximum and minimum for normalization
	float min=1000000;
	float max=-1000000;
	for(int i=0;i<items.size();i++){



		if(items[i][order]<min){
			min=items[i][order];
		}
		if(items[i][order]>max){
			max=items[i][order];
		}
	}
	return max-min;
}

vector<float> algorithm(vector<vector<float> > data,int iteration){ //calculates weights
	vector<float> weights(data[0].size()-1, 0.0);
	for(int i=0;i<iteration;i++){

		vector<int> hit_miss=manhattan(data,i);

		for(int j=0;j<data[0].size()-1;j++){

			weights[j]=weights[j]-(abs(data[i][j]-data[hit_miss[0]][j])/(iteration*difference_low_high(data,j)))+(abs(data[i][j]-data[hit_miss[1]][j])/(iteration*difference_low_high(data,j)));

		}

	}

	return weights;

}

vector<float> vector_to_float(vector<string> x){ //converts vector<string> to vector<float>
	vector<float> instance;
	for (int i=0; i<x.size(); i++)
	{
		double num = stod(x.at(i).c_str());
		instance.push_back(num);
	}
	return instance;
}

vector<int> find_index(vector<float> x,int y){ //finds y maximum elements in x
	vector<int> indexes;
	int maxElementIndex;
	for (int i=0; i<y; i++)
	{
		maxElementIndex = max_element(x.begin(),x.end()) - x.begin();
		x[maxElementIndex]=-10000;
		indexes.push_back(maxElementIndex);


	}
	sort(indexes.begin(), indexes.end());
	return indexes;
}

int main(int argc, char** argv) {
	vector<int> indexes;

	MPI_Init(NULL, NULL);
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);


	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);


	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int name_len;
	MPI_Get_processor_name(processor_name, &name_len);

	if (world_rank == 0) {
		string fileName=argv[1];

		read_tsv(fileName); // only master reads from the file


	}

	for (int j=0; j<world_size-1; j++)
	{
		MPI_Send(&numOfProcessor, 1, MPI_INT, j+1, 0, MPI_COMM_WORLD);
		MPI_Send(&features, 1, MPI_INT, j+1, 0, MPI_COMM_WORLD);

		MPI_Send(&instances, 1, MPI_INT, j+1, 0, MPI_COMM_WORLD);

		MPI_Send(&iterations, 1, MPI_INT, j+1, 0, MPI_COMM_WORLD);
		MPI_Send(&top_features, 1, MPI_INT, j+1, 0, MPI_COMM_WORLD);


	}
	if(world_rank != 0 ){
		MPI_Recv(&numOfProcessor, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&features, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		MPI_Recv(&instances, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&iterations, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&top_features, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	}



	vector<float> instance;
	vector<int> overall_indexes;

	vector<vector<float>> data(instances/(numOfProcessor-1), vector<float> (features+1, 0));

	for (int j=0; j<numOfProcessor-1; j++)
	{
		for (int i=0; i<instances/(numOfProcessor-1); i++)
		{
			if (world_rank == 0) {
				int row=i+j*items.size()/(numOfProcessor-1);
				instance=vector_to_float(items[row]);
				MPI_Send( &instance[0], features+1, MPI_FLOAT, j+1, 0, MPI_COMM_WORLD); //data is sent to other processors
			}
			if (world_rank == j+1){
				instance.resize(features+1);
				MPI_Recv(&instance[0], features+1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); //data is received from master (P0)
				data[i]=instance;
			}
		}
	}




	for (int j=0; j<numOfProcessor-1; j++)
	{

		if (world_rank == 0) {
			indexes.resize(top_features);
			MPI_Recv( &indexes[0], top_features, MPI_INT, j+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			for (int i=0; i<indexes.size(); i++)
			{
				overall_indexes.push_back(indexes[i]);
			}

		}
		if (world_rank == j+1){
			vector<float> weights=algorithm(data,iterations);

			indexes=find_index(weights,top_features);

			printf("Slave P%d :",j+1); //each processor prints the index of most informative features
			for (int j=0; j<indexes.size(); j++)
			{
				printf(" %d",indexes[j]);
			}
			printf("\n");


			MPI_Send(&indexes[0], top_features, MPI_INT, 0, 0, MPI_COMM_WORLD);



		}


	}
	nanosleep(&delta, &delta);

	if (world_rank ==0) {
		sort(overall_indexes.begin(), overall_indexes.end()); //the results gathered from processors are sorted
		overall_indexes.erase(unique( overall_indexes.begin(), overall_indexes.end() ), overall_indexes.end() ); //the duplicates are eliminated


		printf("Master P0 :");
		for (int j=0; j<overall_indexes.size(); j++)
		{
			printf(" %d",overall_indexes[j]);
		}
		printf("\n");




	}

	MPI_Finalize();

}
