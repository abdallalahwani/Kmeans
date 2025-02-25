#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

struct ClusterPoint {
  double sum;
  double mean;
  int size;
};

double find_max(double *data,int size);
void change_data_with_elm(double *data, int size, double elm);
void add_to_data(double *data, double *arr, int start, int end);
double get_euclidean_distance(double *X, struct ClusterPoint *centroid, int size);
void assignToClosestCluster(double *X, struct ClusterPoint *centroids, int size_centroids, int size_X);
int get_index_of_min(double *arr, int size);
int check_euclidean_dist_for_every_centroid(double *prevCentroids, struct ClusterPoint *centroids, int size, int size_centroid);

double find_max(double *data,int size) {
  double max;
  int i;

  max = data[0];
  i = 1;
  while (i < size) {
    if (data[i] > max) {
      max = data[i];
    }
    i += 1;
  }
  return max;
}

void change_data_with_elm(double *data, int size, double elm) {
  int i;

  i = 0;
  while (i < size) {
    data[i] = elm;
    i+= 1;
  }
}

void add_to_data(double *data, double *arr, int start, int end) {
  int i;
  
  i = start;
  while (i < end) {
    data[i] = arr[i - start];
    i+= 1;
  }
}

double get_euclidean_distance(double *X, struct ClusterPoint *centroid, int size) {
  double sum;
  int i;

  sum = 0;
  i = 0;
  while (i < size) {
    sum += pow(X[i] - (centroid[i].mean), 2);
    i += 1;
  }
  return sqrt(sum);
}

int get_index_of_min(double *arr, int size) {
  double min;
  int index;
  int i;

  min = arr[0];
  i = 1;  
  index = 0;
  while (i < size) {
    if(arr[i] < min) {
      min = arr[i];
      index = i;
    }
    i += 1;
  }
  return index;
}

void assignToClosestCluster(double *X, struct ClusterPoint *centroids, int size_centroids, int size_X) {
  double *argminDistValues;
  struct ClusterPoint *centroid;
  double d_X_centroid;
  int closestCluster;
  int start;
  int i;

  argminDistValues = (double*) malloc((int)(size_centroids/size_X) * sizeof(double));
  centroid = (struct ClusterPoint*) malloc(size_X * sizeof(struct ClusterPoint));  
  i = 0;
  while(i < size_centroids) {
    centroid[i%size_X].sum = centroids[i].sum;
    centroid[i%size_X].size = centroids[i].size;
    centroid[i%size_X].mean = centroids[i].mean;
    i += 1;
    if (i % size_X == 0) {
      d_X_centroid = get_euclidean_distance(X, centroid, size_X);
      argminDistValues[(int)(i/size_X)-1] = d_X_centroid;
    }
  }

  i = 0;
  closestCluster = get_index_of_min(argminDistValues, (int)(size_centroids/size_X));
  start = closestCluster*(size_X);
  while (i < size_X) {
    centroids[start + i].sum += X[i];
    centroids[start + i].size += 1;
    i += 1;
  }
  
  free(centroid);
  free(argminDistValues);
}

/*1 means its truthy 0 means its falsy*/ 
int check_euclidean_dist_for_every_centroid(double *prevCentroids, struct ClusterPoint *centroids, int size, int size_centroid) {
  int i;
  double *prevCentroid;
  struct ClusterPoint *centroid;
  double dist;

  i = 0;
  prevCentroid = (double*) malloc(size_centroid * sizeof(double));
  centroid = (struct ClusterPoint*) malloc(size_centroid * sizeof(struct ClusterPoint));
  while (i < size) {
    prevCentroid[i%size_centroid] = prevCentroids[i];
    centroid[i%size_centroid].sum = centroids[i].sum;
    centroid[i%size_centroid].size = centroids[i].size;
    centroid[i%size_centroid].mean = centroids[i].mean;

    i += 1;

    if(i%size_centroid == 0) {
      dist = get_euclidean_distance(prevCentroid, centroid, size_centroid);
      if(dist >= 0.001) {
        free(centroid);
        free(prevCentroid);
        return 1;
      }
    }
  }

  free(centroid);
  free(prevCentroid);
  return 0;
  
}

struct ClusterPoint* fit(struct ClusterPoint* centroids, double* data,int K ,int N, int d, int iter) {
  int j;
  double *prevCentroids; 
  double *centroid;
  double *X;


  /*init prevCentroids as the max value in each centroid*/ 
  prevCentroids = (double*) malloc(K*d * sizeof(double));
  centroid = (double*) malloc(d * sizeof(double));
  j = 0;

  while (j < K*d) {
    centroid[j%d] = centroids[j].mean;
    j += 1;

    if(j%d == 0) {
      double max;
      max = find_max(centroid, d);
      change_data_with_elm(centroid, d, max);
      add_to_data(prevCentroids, centroid, j - d,j);
    }
  }
    
  j = 0;
  X = (double*) malloc(d * sizeof(double));
  while(check_euclidean_dist_for_every_centroid(prevCentroids, centroids, K*d, d) == 1 && (j < iter)){
    int i;
    
    i = 0;
    /*assign every data point to its closest cluster*/ 
    while (i < N*d) {
      X[i%d] = data[i];
      i += 1;
      if(i % d == 0) {
        assignToClosestCluster(X,centroids,K*d,d);
      }
    }

    i = 0;
    /*update the centroids*/ 
    while (i < K*d) {
      prevCentroids[i] = centroids[i].mean;
      centroids[i].mean = (double)(centroids[i].sum) / (double)(centroids[i].size);
      centroids[i].sum = 0;
      centroids[i].size = 0;
      i += 1;
    }

    /*update j*/ 
    j += 1;
  }
/*free all allocated memory*/ 
  free(centroid);
  free(prevCentroids);
  free(X);

  return centroids;
}

static PyObject* get_centroids(PyObject* self, PyObject* args)
{
  PyObject *lst1;
  PyObject *lst2;
  PyObject *item;
  PyObject *python_val;
  int K;
  int d;
  int N;
  int iter;
  int i;
  double point;
  struct ClusterPoint* points;
  
    
  if (!PyArg_ParseTuple(args, "OOiiii", &lst1, &lst2, &K, &N, &d, &iter)) {
    return NULL;
  }
  double *data = (double *) malloc (N*d*sizeof(double));
  if(data == NULL) {
    printf("An Error Has Occurred");
    return NULL;
  }
  struct ClusterPoint *centroids = (struct ClusterPoint*) malloc(K*d * sizeof(struct ClusterPoint));
  if(centroids == NULL) {
    printf("An Error Has Occurred");
    return NULL;
  }
  i = 0;

  // extract centroids
  while(i < K*d) {
    struct ClusterPoint clusterPoint;
    item = PyList_GetItem(lst1, i);
    point = PyFloat_AsDouble(item);
    clusterPoint.mean = point;
    clusterPoint.size = 0;
    clusterPoint.sum = 0;
    centroids[i] = clusterPoint;
    i+= 1;
  }

  i = 0;
  // extract data points
  while (i < N*d) {
    item = PyList_GetItem(lst2, i);
    point = PyFloat_AsDouble(item);
    data[i] = point;
    i += 1;
  }

  // create the points from the fit function
  points = fit(centroids,data,K,N,d,iter);

  i = 0;
  python_val = PyList_New(K*d);
  while(i < K*d) {
    PyList_SetItem(python_val, i, Py_BuildValue("d", points[i].mean));
    i += 1;
  }

  free(centroids);
  free(data);

  return python_val;
}

static PyMethodDef kmeansMehtods[] = {
  {"fit",                   
    (PyCFunction) get_centroids, 
    METH_VARARGS,          
    PyDoc_STR("Calculate kmeans using the passed initial centroids,\n arguments:\n"
              "1. centroids = a list of all the calculated initial centroids, "
              "should be a one dimension list of K cenntroids with d points "
              "(a length of K*d).\n"
              "2 data = a list of all the given points, where each point is represented "
              "as a d consecutive floating numbers in the list, (length of d * N)\n"
              "3. K = the number of the given clusters, which is the length of the given centroids.\n"
              "4. N = the number of the given points in the data field.\n"
              "5. d = the length of a point in the given data and the given centroids.\n")}, 
  {NULL, NULL, 0, NULL}     
};

static struct PyModuleDef kmeansmodule = {
  PyModuleDef_HEAD_INIT,
  "mykmeanssp", 
  NULL, 
  -1,  
  kmeansMehtods 
};

PyMODINIT_FUNC PyInit_mykmeanssp(void)
{
  PyObject *m;
  m = PyModule_Create(&kmeansmodule);
  if (!m) {
    return NULL;
  }
  return m;
}