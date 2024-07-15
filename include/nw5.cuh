#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>

#define NUMBER_OF_THREADS 1024
#define TILE_SIZE 32

#define CHECK(call)                                                                 \
  {                                                                                 \
    const cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                       \
      printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                           \
    }                                                                               \
  }

#define CHECK_KERNELCALL()                                                          \
  {                                                                                 \
    const cudaError_t err = cudaGetLastError();                                     \
    if (err != cudaSuccess) {                                                       \
      printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                           \
    }                                                                               \
  }


__global__ void init_borders_v5(int *d_score, int n, int m, int gap);

__global__ void fill_matrix_v5(int *d_score, const char *d_seq1, const char *d_seq2, int match, int mismatch, int gap, int n, int m);

void nw5(const std::string &seq1, const std::string &seq2, int match, int mismatch, int gap);