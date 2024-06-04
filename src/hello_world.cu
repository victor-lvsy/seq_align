#include <stdio.h>

__global__ void helloCUDA()
{
    printf("Hello, CUDA!\n");
}

void call_kernel(){
    helloCUDA<<<1, 1>>>();
    cudaDeviceSynchronize();
}

