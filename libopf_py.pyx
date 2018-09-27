# code

cimport libopf_py
import numpy as np
cimport numpy as np

d = {
      "euclidian"          : libopf_py.EUCLIDIAN,
      "log_euclidian"      : libopf_py.LOG_EUCLIDIAN,
      "chi_square"         : libopf_py.CHI_SQUARE,
      "manhattan"          : libopf_py.MANHATTAN,
      "canberra"           : libopf_py.CANBERRA,
      "squared_chord"      : libopf_py.SQUARED_CHORD,
      "squared_chi_square" : libopf_py.SQUARED_CHI_SQUARE,
      "bray_curtis"        : libopf_py.BRAY_CURTIS
    }

cdef class OPF:

  cdef libopf_py.opf_graph * sg
  cdef bint supervised
  cdef bint precomputed_distance

  cdef int node_n
  cdef int feat_n
  cdef str metric

  def __cinit__(self):
      self.sg = NULL
      self.node_n = 0
      self.feat_n = 0
      self.supervised = True
      self.precomputed_distance = False
      self.metric = "euclidian"

  def __dealloc__(self):
    if self.sg is not NULL:
      libopf_py.opf_graph_destroy (&self.sg)

  def __reduce__(self):
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] path_val, radius
    cdef np.ndarray[np.float64_t, ndim=2, mode='c'] X
    cdef np.ndarray[np.int32_t,   ndim=1, mode='c'] ordered_list_of_nodes, position, label

    path_val              = np.empty(self.node_n)
    label                 = np.empty(self.node_n, dtype=np.int32)
    ordered_list_of_nodes = np.empty(self.node_n, dtype=np.int32)
    position              = np.empty(self.node_n, dtype=np.int32)
    radius                = np.empty(self.node_n)

    if not self.precomputed_distance:
      X = np.empty((self.node_n, self.feat_n))
    else:
      X = np.empty(0)

    libopf_py.opf_graph_get_fit_data (self.sg,
                                      <double *>path_val.data,
                                      <int *>label.data,
                                      <int *>ordered_list_of_nodes.data,
                                      <int *>position.data,
                                      <double *>radius.data,
                                      <double *>X.data)

    return OPF_unpickle, (self.node_n, self.feat_n, self.metric, self.supervised,
                          self.precomputed_distance, path_val, label, ordered_list_of_nodes,
                          position, radius, X)

  def fit (self,
          np.ndarray[np.float64_t, ndim=2, mode='c'] X,
          np.ndarray[np.int32_t,   ndim=1, mode='c'] Y = None,
          learning="default", metric="euclidian",
          bint precomputed_distance=False, double split=0.8):


      if Y is not None:
        self.supervised = True
      else:
        self.supervised = False

      self.precomputed_distance = precomputed_distance

      self.node_n = <int>X.shape[0]
      self.feat_n = <int>X.shape[1]
      self.metric = metric

      if Y is not None and X.shape[0] != Y.shape[0]:
        raise Exception("Shape mismatch")

      if self.precomputed_distance and X.shape[0] != X.shape[1]:
        raise Exception("Distance matrix should be squared, but it's (%s,%s)" %
                                                     (X.shape[0], X.shape[1]))

      if self.supervised:
        if learning not in ("default", "iterative", "agglomerative"):
          raise Exception("Invalid training mode")

      self.sg = libopf_py.opf_graph_create (<int>X.shape[0])
      if self.sg == NULL:
        raise MemoryError("Seems we've run out of of memory")

      if self.precomputed_distance:
        if self.supervised:
          if not libopf_py.opf_graph_set_precomputed_distance (self.sg,
                                                              <double*>X.data,
                                                              <int*>Y.data):
            raise MemoryError("Seems we've run out of of memory")
        else:
          if not libopf_py.opf_graph_set_precomputed_distance (self.sg,
                                                              <double*>X.data,
                                                              NULL):
            raise MemoryError("Seems we've run out of of memory")

      else:

        if self.supervised:
          if not libopf_py.opf_graph_set_feature (self.sg,
                                                 <double*>X.data,
                                                 <int*>Y.data,
                                                 <int>X.shape[1]):
            raise MemoryError("Seems we've run out of of memory")
        else:
          if not libopf_py.opf_graph_set_feature (self.sg,
                                                 <double*>X.data,
                                                 NULL,
                                                 <int>X.shape[1]):
            raise MemoryError("Seems we've run out of of memory")

        libopf_py.opf_graph_set_metric (self.sg, NULL, d[metric])

      if self.supervised:
        if learning == "default":
          libopf_py.opf_supervised_train (self.sg)
        elif learning == "iterative":
          libopf_py.opf_supervised_train_iterative (self.sg, split)
        elif learning == "agglomerative":
          libopf_py.opf_supervised_train_agglomerative (self.sg, split)
      else:
        libopf_py.opf_best_k_min_cut (self.sg, 1, 10)
        libopf_py.opf_unsupervised_clustering (self.sg)

  def predict(self, np.ndarray[np.float64_t, ndim=2, mode='c'] X):

    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] labels

    if self.precomputed_distance:
      labels = np.empty(X.shape[1], dtype=np.int32)
    else:
      labels = np.empty(X.shape[0], dtype=np.int32)

    if self.supervised == None:
      raise Exception ("Not fitted!")

    if self.precomputed_distance and X.shape[0] != self.node_n:
      raise Exception("Distance matrix shape is wrong")

    if not self.precomputed_distance and X.shape[1] != self.feat_n:
      raise Exception("Feature matrix shape is wrong")

    if self.precomputed_distance:
      if self.supervised:
        libopf_py.opf_supervised_classify (self.sg,
                                           <double*>X.data,
                                           <int>X.shape[1],
                                           <int*>labels.data)
      else:
        libopf_py.opf_unsupervised_knn_classify (self.sg,
                                                 <double*>X.data,
                                                 <int>X.shape[1],
                                                 <int*>labels.data)
    else:
      if self.supervised:
        libopf_py.opf_supervised_classify (self.sg,
                                           <double*>X.data,
                                           <int>X.shape[0],
                                           <int*>labels.data)
      else:
        libopf_py.opf_unsupervised_knn_classify (self.sg,
                                                 <double*>X.data,
                                                 <int>X.shape[0],
                                                 <int*>labels.data)

    return labels

def OPF_unpickle (*args):
  cdef np.ndarray[np.float64_t, ndim=1, mode='c'] path_val, radius
  cdef np.ndarray[np.float64_t, ndim=2, mode='c'] X
  cdef np.ndarray[np.int32_t,   ndim=1, mode='c'] Y, ordered_list_of_nodes, position
  cdef OPF opf

  node_n, feat_n, metric, supervised, precomputed_distance, \
  path_val, Y, ordered_list_of_nodes, position, \
  radius, X = args

  opf = OPF()

  opf.node_n = node_n
  opf.feat_n = feat_n
  opf.metric = metric
  opf.supervised = supervised
  opf.precomputed_distance = precomputed_distance

  opf.sg = libopf_py.opf_graph_create (opf.node_n)

  if opf.precomputed_distance:
    if opf.supervised:
      if not libopf_py.opf_graph_set_precomputed_distance (opf.sg,
                                                          <double*>X.data,
                                                          <int*>Y.data):
        raise MemoryError("Seems we've run out of of memory")
    else:
      if not libopf_py.opf_graph_set_precomputed_distance (opf.sg,
                                                          <double*>X.data,
                                                          NULL):
        raise MemoryError("Seems we've run out of of memory")

  else:

    if opf.supervised:
      if not libopf_py.opf_graph_set_feature (opf.sg,
                                             <double*>X.data,
                                             <int*>Y.data,
                                             <int>X.shape[1]):
        raise MemoryError("Seems we've run out of of memory")
    else:
      if not libopf_py.opf_graph_set_feature (opf.sg,
                                             <double*>X.data,
                                             NULL,
                                             <int>X.shape[1]):
        raise MemoryError("Seems we've run out of of memory")

    libopf_py.opf_graph_set_metric (opf.sg, NULL, d[metric])

  libopf_py.opf_graph_set_fit_data (opf.sg,
                                    <double *>path_val.data,
                                    <int *>Y.data,
                                    <int *>ordered_list_of_nodes.data,
                                    <int *>position.data,
                                    <double *>radius.data)

  return opf

