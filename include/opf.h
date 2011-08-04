#ifndef _OPF_H_
#define _OPF_H_

#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <assert.h>
#include <sys/time.h>
#include <time.h>

#include "common.h"
#include "set.h"
#include "gqueue.h"
#include "subgraph.h"
#include "sgctree.h"
#include "realheap.h"

/*--------- Common definitions --------- */
#define opf_MAXARCW			100000.0
#define opf_MAXDENS			1000.0
#define opf_PROTOTYPE		1

typedef float (*opf_ArcWeightFun) (float *f1, float *f2, int n);

extern opf_ArcWeightFun opf_ArcWeight;

extern char opf_PrecomputedDistance;
extern float **opf_DistanceValue;

/*--------- Supervised OPF with complete graph -----------------------*/
void opf_OPFTraining (Subgraph * sg);   //Training function
void opf_OPFClassifying (Subgraph * sgtrain, Subgraph * sg);    //Classification function: it simply classifies samples from sg
void opf_OPFLearning (Subgraph ** sgtrain, Subgraph ** sgeval); //Learning function
void opf_OPFAgglomerativeLearning (Subgraph ** sgtrain, Subgraph ** sgeval);    //Agglomerative learning function

/*--------- Unsupervised OPF -------------------------------------*/
void opf_OPFClustering (Subgraph * sg); //Training function: it computes unsupervised training for the pre-computed best k.

void opf_OPFKNNClassify (Subgraph * sgtrain, Subgraph * sg);    // Classification function: it classifies nodes of sg by using the labels created by program opf_cluster in sgtrain

/*------------ Auxiliary functions ------------------------------ */
void opf_ResetSubgraph (Subgraph * sg); //Resets subgraph fields (pred and arcs)
void opf_SwapErrorsbyNonPrototypes (Subgraph ** sgtrain, Subgraph ** sgeval);   //Replace errors from evaluating set by non prototypes from training set
void opf_MoveIrrelevantNodes (Subgraph ** sgtrain, Subgraph ** sgeval); //Move irrelevant nodes from source graph (src) to destiny graph (dst)
void opf_MoveMisclassifiedNodes (Subgraph ** sgtrain, Subgraph ** sgeval, int *p);      //Move misclassified nodes from source graph (src) to destiny graph (dst)
void opf_RemoveIrrelevantNodes (Subgraph ** sg);        //Remove irrelevant nodes
void opf_MarkNodes (Subgraph * g, int i);       //mark nodes and the whole path as relevants
void opf_WriteModelFile (Subgraph * g, char *file);     //write model file to disk
Subgraph *opf_ReadModelFile (char *file);       //read subgraph from opf model file
void opf_NormalizeFeatures (Subgraph * sg);     //normalize features
void opf_MSTPrototypes (Subgraph * sg); //Find prototypes by the MST approach
Subgraph **opf_kFoldSubgraph (Subgraph * sg, int k);    //It creates k folds for cross validation
void opf_SplitSubgraph (Subgraph * sg, Subgraph ** sg1, Subgraph ** sg2, float perc1);  //Split subgraph into two parts such that the size of the first part  is given by a percentual of samples.
Subgraph *opf_MergeSubgraph (Subgraph * sg1, Subgraph * sg2);   //Merge two subgraphs
float opf_Accuracy (Subgraph * g);      //Compute accuracy
int **opf_ConfusionMatrix (Subgraph * sg);      //Compute the confusion matrix
float **opf_ReadDistances (char *fileName, int *n);     //read distances from precomputed distances file
float opf_NormalizedCut (Subgraph * sg);
void opf_BestkMinCut (Subgraph * sg, int kmin, int kmax);
void opf_CreateArcs (Subgraph * sg, int knn);   //it creates arcs for each node (adjacency relation)
void opf_DestroyArcs (Subgraph * sg);   //it destroys the adjacency relation
void opf_PDF (Subgraph * sg);   //it computes the PDf for each node
void opf_ElimMaxBelowVolume (Subgraph * sg, int V);     // Eliminate maxima in the graph with volume below V
void opf_ElimMaxBelowArea (Subgraph * sg, int A);       //Eliminate maxima in the graph with area below A
void opf_ElimMaxBelowH (Subgraph * sg, float H);        // Eliminate maxima in the graph with pdf below H

/*------------ Distance functions ------------------------------ */
float opf_EuclDist (float *f1, float *f2, int n);       //Computes Euclidean distance between feature vectors
float opf_EuclDistLog (float *f1, float *f2, int n);    //Computes Euclidean distance between feature vectors and applies
                                                    //logarithmic function
float opf_GaussDist (float *f1, float *f2, int n, float gamma); //Computes Gaussian distance between feature vectors
float opf_ChiSquaredDist (float *f1, float *f2, int n); //Compute  chi-squared distance between feature vectors
float opf_ManhattanDist (float *f1, float *f2, int n);  //Compute  Manhattan distance between feature vectors
float opf_CanberraDist (float *f1, float *f2, int n);   //Compute  Camberra distance between feature vectors
float opf_SquaredChordDist (float *f1, float *f2, int n);       //Compute  Squared Chord distance between feature vectors
float opf_SquaredChiSquaredDist (float *f1, float *f2, int n);  //Compute  Squared Chi-squared distance between feature vectors
float opf_BrayCurtisDist (float *f1, float *f2, int n); //Compute  Bray Curtis distance between feature vectors

/* -------- Auxiliary functions used to optimize BestkMinCut -------- */
float *opf_CreateArcs2 (Subgraph * sg, int kmax);       //Creates arcs for each node (adjacency relation) and returns
                                               //the maximum distances for each k=1,2,...,kmax
void opf_OPFClusteringToKmax (Subgraph * sg);   //OPFClustering computation only for sg->bestk neighbors
void opf_PDFtoKmax (Subgraph * sg);     //PDF computation only for sg->bestk neighbors
float opf_NormalizedCutToKmax (Subgraph * sg);  //Normalized cut computed only for sg->bestk neighbors
#endif
