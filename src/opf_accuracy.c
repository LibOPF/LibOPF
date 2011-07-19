#include "OPF.h"

int main(int argc, char **argv){
	fflush(stdout);
	fprintf(stdout, "\nProgram that computes OPF accuracy of a given set\n");
	fprintf(stdout, "\nIf you have any problem, please contact: ");
	fprintf(stdout, "\n- alexandre.falcao@gmail.com");
	fprintf(stdout, "\n- papa.joaopaulo@gmail.com\n");
	fprintf(stdout, "\nLibOPF version 2.0 (2009)\n");
	fprintf(stdout, "\n"); fflush(stdout);

	if(argc != 2){
		fprintf(stderr, "\nusage opf_accuracy <P1>");
		fprintf(stderr, "\nP1: data set in the OPF file format");
		exit(-1);
	}

	int i;
	float Acc;
	size_t result;
	FILE *f = NULL;
	char fileName[256];

	fprintf(stdout, "\nReading data file ..."); fflush(stdout);
	Subgraph *g = ReadSubgraph(argv[1]);
	fprintf(stdout, " OK"); fflush(stdout);

	fprintf(stdout, "\nReading output file ..."); fflush(stdout);
	sprintf(fileName,"%s.out",argv[1]);
	f = fopen(fileName,"r");
	if(!f){
		fprintf(stderr,"\nunable to open file %s", argv[2]);
		exit(-1);
	}
	for (i = 0; i < g->nnodes; i++)
		result = fscanf(f,"%d",&g->node[i].label);
	fclose(f);
	fprintf(stdout, " OK"); fflush(stdout);

	fprintf(stdout, "\nComputing accuracy ..."); fflush(stdout);
	Acc = opf_Accuracy(g);
	fprintf(stdout, "\nAccuracy: %.2f%%", Acc*100); fflush(stdout);

	fprintf(stdout, "\nWriting accuracy in output file ..."); fflush(stdout);
	sprintf(fileName,"%s.acc",argv[1]);
	f = fopen(fileName,"a");
	result = fprintf(f,"%f\n",Acc*100);
	fclose(f);
	fprintf(stdout, " OK"); fflush(stdout);

	fprintf(stdout, "\nDeallocating memory ..."); fflush(stdout);
	DestroySubgraph(&g);
	fprintf(stdout, " OK\n");

	return 0;
}
