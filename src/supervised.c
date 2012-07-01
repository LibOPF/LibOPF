/*
  Copyright (C) <2009-2011> <Alexandre Xavier Falcão and João Paulo Papa>

  Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
  Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

  please see full copyright in COPYING file.

  -------------------------------------------------------------------------
  written by A.X. Falcão <afalcao@ic.unicamp.br> and by J.P. Papa
  <papa.joaopaulo@gmail.com>, Oct 20th 2008

  This program is a collection of functions to manage the Optimum-Path Forest (OPF)
  classifier.*/

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
#include "graph.h"
#include "realheap.h"
#include "metrics.h"
#include "measures.h"
#include "knn.h"
#include "supervised.h"

#include <omp.h>

// Find prototypes by the MST approach
static void
mst_prototypes (struct opf_graph * sg)
{
  int p, q;
  double weight;
  struct real_heap *Q = NULL;
  double *path_val = NULL;
  int pred;
  int nproto;

  // initialization
  path_val = alloc_double (sg->node_n);
  Q = real_heap_create (sg->node_n, path_val);

  for (p = 0; p < sg->node_n; p++)
    {
      path_val[p] = DBL_MAX;
      sg->node[p].status = STATUS_NOTHING;
    }

  path_val[0] = 0;
  sg->node[0].pred = NIL;
  real_heap_insert (Q, 0);

  nproto = 0;

  // Prim's algorithm for Minimum Spanning Tree
  while (!real_heap_is_empty (Q))
    {
      real_heap_remove (Q, &p);
      assert (p >= 0 && p < sg->node_n);

      sg->node[p].path_val = path_val[p];

      pred = sg->node[p].pred;
      if (pred != NIL)
        if (sg->node[p].label_true != sg->node[pred].label_true)
          {
            if (sg->node[p].status != STATUS_PROTOTYPE)
              {
                sg->node[p].status = STATUS_PROTOTYPE;
                nproto++;
              }
            if (sg->node[pred].status != STATUS_PROTOTYPE)
              {
                sg->node[pred].status = STATUS_PROTOTYPE;
                nproto++;
              }
          }

      for (q = 0; q < sg->node_n; q++)
        {
          if (Q->color[q] != COLOR_BLACK)
            {
              if (p != q)
                {
                  weight = opf_graph_get_distance (sg, &sg->node[p], &sg->node[q]);

                  if (weight < path_val[q])
                    {
                      sg->node[q].pred = p;
                      real_heap_update (Q, q, weight);
                    }
                }
            }
        }
    }
  real_heap_destroy (&Q);
  free (path_val);

  /* the algorithm will work even if there
     is just one class in the training set */
  if (nproto == 0)
    sg->node[0].status = STATUS_PROTOTYPE;
}

void
opf_supervised_train (struct opf_graph * sg)
{
  int p, q, i;
  double tmp, weight;
  struct real_heap *Q = NULL;
  double *path_val = NULL;

  // compute optimum prototypes
  mst_prototypes (sg);

  // initialization
  path_val = alloc_double (sg->node_n);

  Q = real_heap_create (sg->node_n, path_val);

  for (p = 0; p < sg->node_n; p++)
    {
      if (sg->node[p].status == STATUS_PROTOTYPE)
        {
          sg->node[p].pred = NIL;
          path_val[p] = 0;
          sg->node[p].label = sg->node[p].label_true;
          real_heap_insert (Q, p);
        }
      else // non-prototypes
        {
          path_val[p] = DBL_MAX;
        }
    }

  // IFT with fmax
  i = 0;
  while (!real_heap_is_empty (Q))
    {
      real_heap_remove (Q, &p);
      assert (p >= 0 && p < sg->node_n);

      sg->ordered_list_of_nodes[i] = p;

      i++;
      sg->node[p].path_val = path_val[p];

      for (q = 0; q < sg->node_n; q++)
        {
          if (p != q)
            {
              if (path_val[p] < path_val[q])
                {
                  weight = opf_graph_get_distance (sg, &sg->node[p], &sg->node[q]);

                  tmp = MAX (path_val[p], weight);
                  if (tmp < path_val[q])
                    {
                      sg->node[q].pred = p;
                      sg->node[q].label = sg->node[p].label;
                      real_heap_update (Q, q, tmp);
                    }
                }
            }
        }
    }

  real_heap_destroy (&Q);
  free (path_val);
}


//Classification function: it simply classifies samples from sg -----
void
opf_supervised_classify (struct opf_graph * sg_train, double *feat, int sample_n, int *label)
{
  int i;

  omp_set_num_threads(NTHREADS);
  #pragma omp parallel for
  for (i = 0; i < sample_n; i++)
    {
      int c_label = -1;
      int j = 0;
      double minCost = DBL_MAX;

      for (j=0;

           (j < sg_train->node_n)
           && (minCost > sg_train->node[sg_train->ordered_list_of_nodes[j]].path_val);

          j++)
        {
          int l;
          double tmp, weight;

          l = sg_train->ordered_list_of_nodes[j];
          assert (l >= 0 && l < sg_train->node_n);

          if (sg_train->pdist) /* feat is a distance matrix */
            weight = feat[sg_train->node[l].position*sample_n+i];
          else                 /* feat are feature vectors */
            weight = sg_train->arc_weight
                       (sg_train->node[l].feat, &feat[i*sg_train->feat_n], sg_train->feat_n);

          tmp = MAX (sg_train->node[l].path_val, weight);
          if (tmp < minCost)
            {
              minCost = tmp;
              c_label = sg_train->node[l].label;
            }
        }

      label[i] = c_label;
    }
}


static void
supervised_classify_opf_graph (struct opf_graph * sg_train, struct opf_graph * sg_eval)
{
  int i;

  assert (sg_train->pdist == sg_eval->pdist);
  assert (sg_train->feat_n == sg_eval->feat_n);
  assert (sg_train->arc_weight == sg_eval->arc_weight);

  omp_set_num_threads(NTHREADS);
  #pragma omp parallel for
  for (i = 0; i < sg_eval->node_n; i++)
    {
      int c_label = -1;
      int j = 0;
      double minCost = DBL_MAX;

      for (j=0;

           (j < sg_train->node_n)
           && (minCost > sg_train->node[sg_train->ordered_list_of_nodes[j]].path_val);

          j++)
        {
          int l;
          double tmp, weight;

          l = sg_train->ordered_list_of_nodes[j];
          assert (l >= 0 && l < sg_train->node_n);

          weight = opf_graph_get_distance (sg_train, &sg_train->node[l], &sg_eval->node[i]);

          tmp = MAX (sg_train->node[l].path_val, weight);
          if (tmp < minCost)
            {
              minCost = tmp;
              c_label = sg_train->node[l].label;
            }
        }

      sg_eval->node[i].label = c_label;
    }
}

/*
  OPF Accuracy Function
 */
static double
accuracy (struct opf_graph *sg)
{
  int i;
  int *fp = (int*) malloc(sizeof(int)*sg->label_n);
  int *fn = (int*) malloc(sizeof(int)*sg->label_n);
  int *n  = (int*) malloc(sizeof(int)*sg->label_n);
  memset(fp, 0, sizeof(int)*sg->label_n);
  memset(fn, 0, sizeof(int)*sg->label_n);
  memset(n,  0, sizeof(int)*sg->label_n);

  double sum_e = 0.0;

  for (i=0; i < sg->node_n; i++)
    {
      /* false positives */
      fp[sg->node[i].label]      += (sg->node[i].label != sg->node[i].label_true);
      /* false negatives */
      fn[sg->node[i].label_true] += (sg->node[i].label != sg->node[i].label_true);
      /* number of elements */
      n[sg->node[i].label_true] ++;
    }

  for (i=0; i < sg->label_n; i++)
    {
      /* e_i,1 + e_i,2 */
      sum_e += (double)(fp[i]) / (sg->node_n - n[i]) + (double)(fn[i]) / n[i];
    }

  return 1.0 - sum_e / (2.0 * sg->label_n);
}

/* Replace errors from evaluating set by non prototypes from training set */
static void
swap_wrong_prototypes (struct opf_graph *sg_train, struct opf_graph *sg_eval)
{
  int i;
  int nonprototypes = 0;

  for (i = 0; i < sg_train->node_n; i++)
    if (sg_train->node[i].status == STATUS_NOTHING) nonprototypes++;

  while (i < sg_eval->node_n && nonprototypes > 0)
    {
      if (sg_eval->node[i].label != sg_eval->node[i].label_true)
        {
          /* XXX: this can take a lot of time */
          int j;
          do
            {
              j = random_int (0, sg_train->node_n);
            }
          while (sg_train->node[j].status == STATUS_PROTOTYPE);

          {
            struct snode tmp  = sg_train->node[j];
            sg_train->node[j] = sg_eval->node[i];
            sg_eval->node[i]  = tmp;
          }

          sg_train->node[j].pred = NIL;
          sg_train->node[j].status = STATUS_PROTOTYPE;

          nonprototypes--;
        }
      i++;
    }
}

/* create two pointers to a opf_graph data,
   this function is used in some training modes to reorganize nodes */
static void
opf_graph_split_mirrored (struct opf_graph * sg, double split,
                         struct opf_graph * sg1, struct opf_graph * sg2)
{
  memset (sg1, 0xFF, sizeof(struct opf_graph));
  memset (sg2, 0xFF, sizeof(struct opf_graph));

  sg1->node_n = sg->node_n * split;
  sg2->node_n = sg->node_n - sg1->node_n;

  sg1->node = &sg->node[0];
  sg2->node = &sg->node[sg1->node_n];

  sg1->ordered_list_of_nodes = &sg->ordered_list_of_nodes[0];
  sg2->ordered_list_of_nodes = &sg->ordered_list_of_nodes[sg1->node_n];

  sg1->arc_weight = sg2->arc_weight = sg->arc_weight;
  sg1->feat_data  = sg2->feat_data  = sg->feat_data;
  sg1->feat_n     = sg2->feat_n     = sg->feat_n;

  sg1->pdist                 = sg2->pdist                 = sg->pdist;
  sg1->pdist_train_stride    = sg2->pdist_train_stride    = sg->pdist_train_stride;
}

#define ITER_MAX 10

void
opf_supervised_train_iterative (struct opf_graph *sg, double split)
{
  int i = 0;
  double acc = DBL_MIN, acc_prev = DBL_MIN, delta;

  struct opf_graph sg_train, sg_eval;
  opf_graph_split_mirrored (sg, split, &sg_train, &sg_eval);

  do
    {
      acc_prev = acc;

      opf_supervised_train (&sg_train);
      supervised_classify_opf_graph (&sg_train, &sg_eval);
      acc = accuracy (&sg_eval);

      swap_wrong_prototypes (&sg_train, &sg_eval);

      delta = fabs(acc-acc_prev);
      i++;
    }
  while ((delta > DBL_EPSILON) && (i < ITER_MAX));

  /* just the training part will remain */
  opf_graph_resize (sg, sg_train.node_n);
}

/* Move misclassified data from eval to sg */
static void
move_misclassified_nodes (struct opf_graph *sg_train, struct opf_graph *sg_eval, int *n)
{
  int i;
  int misclassified_n = 0;

  /* count number of misclassied samples in eval */
  for (i=0; i < sg_eval->node_n; i++)
    (sg_eval->node[i].label != sg_eval->node[i].label_true)?
      misclassified_n++ : 0;

  *n = misclassified_n;

  /* nothing to do */
  if (misclassified_n == 0)
    return;

  /* move wrong labelled samples from sg_eval to sg_train */
  for (i=0; i < sg_eval->node_n; i++)
    {
      if (sg_eval->node[i].label != sg_eval->node[i].label_true)
        {
          /* have in mind that sg_train and sg_eval
             are mirrorred to the same opf_graph, so
             sg_train->node[sg_train->node_n] == sg_eval->node[0] */
          {
            struct snode tmp = sg_train->node[sg_train->node_n];
            sg_train->node[sg_train->node_n] = sg_eval->node[i];
            sg_eval->node[i] = tmp;
          }

          sg_train->node_n++;
          sg_eval->node_n--;

          /* sg_train took the first element of sg_eval */
          sg_eval->node++;
          sg_eval->ordered_list_of_nodes++;
        }
    }
}

void
opf_supervised_train_agglomerative (struct opf_graph *sg, double split)
{
    int n;
    struct opf_graph sg_train, sg_eval;
    opf_graph_split_mirrored (sg, split, &sg_train, &sg_eval);

    /* while there exists misclassified samples in eval */
    do
      {
        n = 0;
        opf_supervised_train (&sg_train);
        supervised_classify_opf_graph (&sg_train, &sg_eval);
        move_misclassified_nodes (&sg_train, &sg_eval, &n);
      }
    while(n > 0);

  /* just the training part will remain */
  opf_graph_resize (sg, sg_train.node_n);
}
