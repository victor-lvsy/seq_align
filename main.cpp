#include <cuda_runtime.h>
#include <fstream>
#include <time.h>
#include "include/nw.h"
#include "include/nw1.cuh"

using namespace std;


int  main( int argc, char ** argv )
{
        // Sequences to be aligned
        char  *seq_1;
        char  *seq_2; 

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

            seq_1 = (char *) malloc( 2 * size1 * sizeof(char));
            
            file1 >> seq_1 ;
            
            file1.close();
            printf("Seq 1: %s size1: %d\n",seq_1,size1);
        }
        

        ifstream file2("./test_files/test2.txt");
        if (file2.fail())
             perror ("Error opening file 2");
        else
        {
            file2.seekg (0, ios::end);
            size2 = file2.tellg();
            file2.seekg (0, ios::beg);
            seq_2 = (char *) malloc(  2 * size2 * sizeof(char));
            
            file2 >> seq_2 ;
            
            file2.close();
            printf("Seq 2: %s size2: %d\n",seq_2,size2);
        }
        



    struct timespec t1,t2; double dt1;
    clock_gettime(CLOCK_REALTIME,  &t1);

        const std::string seq_1_std = seq_1;
        const std::string seq_2_std = seq_2;

        // Get alignment
        nw(seq_1_std, seq_2_std, 1, -1, -1) ;   


    clock_gettime(CLOCK_REALTIME,  &t2);
    dt1 = (t2.tv_sec - t1.tv_sec)  + (double) (t2.tv_nsec - t1.tv_nsec) * 1e-9  ;
    double time=dt1*1000 ;

    nw1(seq_1_std, seq_2_std, 1, -1, -1);

    printf("\n\n%10f kernel Time elapsed \n", time);

    return  0 ;
}
























