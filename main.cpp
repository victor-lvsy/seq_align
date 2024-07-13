#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include "include/nw.h"
#include "include/nw1.cuh"
#include "include/nw4.cuh"
#include "include/nw3.cuh"
#include "include/nw2.cuh"


using namespace std;

void printDeviceProperties() {
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        std::cerr << "Failed to get device count: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    std::cout << "Number of CUDA devices: " << deviceCount << std::endl;

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, i);

        if (err != cudaSuccess) {
            std::cerr << "Failed to get properties for device " << i << ": " << cudaGetErrorString(err) << std::endl;
            continue;
        }

        std::cout << "Device " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Multiprocessor Count: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Max Threads Per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Max Threads Dim: [" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << "]" << std::endl;
        std::cout << "  Max Grid Size: [" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << "]" << std::endl;
        std::cout << std::endl;
    }
}


int  main( int argc, char ** argv )
{
        // Sequences to be aligned
        vector<char> seq_1;
        vector<char> seq_2;

        //FILE  *file1 , *file2 ;
        int   size1 , size2 ;
        ifstream file1("./test_files/test1.txt");
        if (file1.fail())
             perror ("Error opening file 1");
        else
        {
            file1.seekg (0, ios::end);
            size1 = file1.tellg();
            file1.seekg (0, ios::beg);

            seq_1.resize(2 * size1);
            file1.read(&seq_1[0], size1);
            
            file1.close();
            printf("Seq 1 of size: %d\n",size1);
        }
        

        ifstream file2("./test_files/test2.txt");
        if (file2.fail())
             perror ("Error opening file 2");
        else
        {
            file2.seekg (0, ios::end);
            size2 = file2.tellg();
            file2.seekg (0, ios::beg);
            seq_2.resize(2 * size2); // resizing the vector
            file2.read(&seq_2[0], size2);
            
            file2.close();
            printf("Seq 2 of size: %d\n",size2);
        }
        



    struct timespec t1,t2; double dt1;
    clock_gettime(CLOCK_REALTIME,  &t1);

        const std::string seq_1_std(seq_1.begin(), seq_1.begin() + size1);
        const std::string seq_2_std(seq_2.begin(), seq_2.begin() + size2);

        // Get alignment
        // nw(seq_1_std, seq_2_std, 1, -1, -1) ;   


    clock_gettime(CLOCK_REALTIME,  &t2);
    dt1 = (t2.tv_sec - t1.tv_sec)  + (double) (t2.tv_nsec - t1.tv_nsec) * 1e-9  ;
    double time=dt1*1000 ;

    printDeviceProperties();

    // nw4(seq_1_std, seq_2_std, 1, -1, -1);

    // nw2(seq_1_std, seq_2_std, 1, -1, -1);

    nw1(seq_1_std, seq_2_std, 1, -1, -1);

    printf("\n\n%10f kernel Time elapsed \n", time);

    return  0 ;
}
























