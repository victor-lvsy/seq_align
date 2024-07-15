#include "nw6.cuh"

// Dynamic kernel launches to maximize gpu usage. 
// Disclaimer: This is a simplifed version of the algorithm because longer sequences are split 
// and processed without codependances.

__global__ void kernel_launcher(int *d_score, char *d_seq1, char *d_seq2, int match, int mismatch, int gap, int n, int m){
    int tid = threadIdx.x, number_of_subthreads = NUMBER_OF_THREADS;
    int *test = d_score + tid * (number_of_subthreads * number_of_subthreads);
    char *seq1 = d_seq1 + tid * number_of_subthreads;
    char *seq2 = d_seq2 + tid * number_of_subthreads;

    init_borders_v6<<<1, number_of_subthreads>>>(test, number_of_subthreads, gap);
    fill_matrix_v6<<<1, number_of_subthreads>>>(test, seq1, seq2, match, mismatch, gap, number_of_subthreads - 1, number_of_subthreads - 1);
}

// Kernel for initializing borders of the score matrix
__global__ void init_borders_v6(int *d_score, int n, int gap)
{
    int idx = threadIdx.x;
    int total_diagonals = n;
    
    // Initialize diagonals for the border elements
    if (idx < total_diagonals) {
        if (idx <= n) {
            // Initialize first column elements
            int diag_index = idx;
            int diag_position = 0;
            int pos = diag_index * (diag_index + 1) / 2 + diag_position;
            d_score[pos] = idx * gap; // Setting gap penalties for the first column
        }

        if (idx <= n) {
            // Initialize first row elements
            int diag_index = idx;
            int diag_position = idx;
            int pos = diag_index * (diag_index + 1) / 2 + diag_position;
            d_score[pos] = idx * gap; // Setting gap penalties for the first row
        }
    }
}

// Kernel for filling the matrix using the anti-diagonal approach
__global__ void fill_matrix_v6(int *d_score, const char *d_seq1, const char *d_seq2, int match, int mismatch, int gap, int n, int m) {
    const int total_diagonals = n + m - 1;
    const int tid = threadIdx.x;
    int first_i, last_i, elements_in_diag, i, j;
    int idx = 4, mem = 2, mem2 = 1;
    
    for (int diag = 2; diag <= total_diagonals + 1; ++diag) {
        first_i = max(1, diag - m); // Calculate start i index for the current diagonal
        last_i = min(n, diag - 1);  // Calculate end i index for the current diagonal
        elements_in_diag = last_i - first_i + 1; // Number of elements in the current diagonal

        if (tid < elements_in_diag) {
            i = last_i - tid; // Mapping tid to i
            j = diag - i; // Mapping i to j

            // Create the linear indexes related to the antidiagonal memory accesses.
            int linear_index = idx + tid;
            int linear_index_l = mem + tid;
            int linear_index_t = (diag <= n) ? linear_index_l - 1 : linear_index_l + 1;
            int linear_index_tl = (diag <= n + 1) ? mem2 + tid - 1 : ((diag == n + 2) ? mem2 + tid : mem2 + tid + 1 );
            if (linear_index <= (n + 1) * (m + 1)) {
                int match_score = (d_seq1[i - 1] == d_seq2[j - 1]) ? match : mismatch;
                d_score[linear_index] = max(
                    d_score[linear_index_tl] + match_score,
                    max(
                        d_score[linear_index_l] + gap,
                        d_score[linear_index_t] + gap));
            }
        }
        mem2 = mem;
        mem = idx;
        idx += elements_in_diag;
        if(diag < n){
            idx+=2;
        }
        if(diag == n){
            idx+=1;
        }
        __syncthreads();
    }
}

// Function to convert a diagonal-major format matrix to row-major format
void convertDiagonalToRowMajor3(int* diagMajorMatrix, int n, int* rowMajorMatrix) {
    n -= 1;
    int total_diagonals = 2*n - 1;
    int idx = 0;
    for (int diag = 0; diag <= total_diagonals + 1; ++diag) {
        int first_i = max(0, diag - n); // Calculate start i index for the current diagonal
        int last_i = min(n, diag);  // Calculate end i index for the current diagonal
        int elements_in_diag = last_i - first_i; // Number of elements in the current diagonal
        if(diag > n){
            for(int i = 0; i <= elements_in_diag; i++){
                rowMajorMatrix[(n - i) * (n+1) + (diag - (n - i))] = diagMajorMatrix[idx + i];
            }
        }
        else{
            for(int i = 0; i <= elements_in_diag; i++){
                rowMajorMatrix[(i * (n+1)) + (elements_in_diag - i)] = diagMajorMatrix[idx + i];
            }
        }
        
        idx += elements_in_diag + 1;

    }
}


// Host function for Needleman-Wunsch algorithm
void nw6(const std::string &seq1, const std::string &seq2, int match, int mismatch, int gap)
{
    printf("HELLO THERE, I am nw4\n");

    int n = seq1.length();
    int m = seq2.length();

    printf("Size of seq1: %d\nSize of seq2: %d\nProduct: %d\n", n, m, n * m);

    // Allocate the matrix with padding (extra row and column) on the host
    int *h_score = new int[(n + 1) * (m + 1)];
    int *h_score_r = new int[(n + 1) * (m + 1)];
    std::fill(h_score, h_score + (n + 1) * (m + 1), 0); // Initialize the matrix with zeros

    int *d_score;
    char *d_seq1, *d_seq2;

    // Allocate memory on GPU for score matrix and sequences
    CHECK(cudaMalloc(&d_score, (n + 1) * (m + 1) * sizeof(int)));
    CHECK(cudaMalloc(&d_seq1, n * sizeof(char)));
    CHECK(cudaMalloc(&d_seq2, m * sizeof(char)));

    // Copy sequences from host to GPU
    CHECK(cudaMemcpy(d_seq1, seq1.data(), n * sizeof(char), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_seq2, seq2.data(), m * sizeof(char), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_score, h_score, (n + 1) * (m + 1) * sizeof(int), cudaMemcpyHostToDevice));

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    int num_threads = (n + (NUMBER_OF_THREADS - 1)) / NUMBER_OF_THREADS;
    printf("Number of Threads: %d\n", num_threads);
    kernel_launcher<<<1, num_threads,0,stream1>>>(d_score, d_seq1, d_seq2, match, mismatch, gap, n, m);
    kernel_launcher<<<1, num_threads,0,stream2>>>(d_score, d_seq1, d_seq2, match, mismatch, gap, n, m);
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);


    // Initialize borders of the matrix on GPU
    // int num_threads = max(n, m);
    // int num_blocks = (num_threads + (NUMBER_OF_THREADS - 1)) / NUMBER_OF_THREADS;
    // printf("Number of blocks: %d\nNumber of Threads: %d\n", num_blocks, num_threads);
    // init_borders_v6<<<num_blocks, NUMBER_OF_THREADS>>>(d_score, n, m, gap);
    // CHECK_KERNELCALL();
    // CHECK(cudaDeviceSynchronize());


    // Fill the matrix on GPU
    // fill_matrix_v6<<<num_blocks, NUMBER_OF_THREADS>>>(d_score, d_seq1, d_seq2, match, mismatch, gap, n, m);
    // CHECK_KERNELCALL();
    // CHECK(cudaDeviceSynchronize());
        

    // Copy score matrix back to CPU from GPU
    CHECK(cudaMemcpy(h_score, d_score, (n + 1) * (m + 1) * sizeof(int), cudaMemcpyDeviceToHost));

    // Clean up GPU memory
    CHECK(cudaFree(d_score));
    CHECK(cudaFree(d_seq1));
    CHECK(cudaFree(d_seq2));

    // Print score matrix for debugging
    // for (int i = 0; i <= (m+1)*(n+1); ++i) {
    //     std::cout << h_score[i] << " ";
    // }

    // int *h_score2 = h_score + NUMBER_OF_THREADS*NUMBER_OF_THREADS;

    // Print score matrix for debugging
    // convertDiagonalToRowMajor3(h_score,NUMBER_OF_THREADS,h_score_r);

    // std::cout << std::endl;
    // std::cout << std::endl;

    // Print the row-major matrix
    // for (int i = 0; i < NUMBER_OF_THREADS; ++i) {
    //     for (int j = 0; j < NUMBER_OF_THREADS; ++j) {
    //         std::cout << h_score_r[i * (NUMBER_OF_THREADS) + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // std::cout << std::endl;
    // std::cout << std::endl;

    // convertDiagonalToRowMajor3(h_score2,NUMBER_OF_THREADS,h_score_r);

    // for (int i = 0; i < NUMBER_OF_THREADS; ++i) {
    //     for (int j = 0; j < NUMBER_OF_THREADS; ++j) {
    //         std::cout << h_score_r[i * (NUMBER_OF_THREADS) + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // Backtracking
    // int i = n;
    // int j = m;
    // std::string aligned_seq1, aligned_seq2;

    // while (i > 0 && j > 0) {
    //     if (h_score_r[i * (m + 1) + j] == h_score_r[(i - 1) * (m + 1) + (j - 1)] + (seq1[i - 1] == seq2[j - 1] ? match : mismatch)) {
    //         aligned_seq1 = seq1[i - 1] + aligned_seq1;
    //         aligned_seq2 = seq2[j - 1] + aligned_seq2;
    //         --i;
    //         --j;
    //     } else if (h_score_r[i * (m + 1) + j] == h_score_r[(i - 1) * (m + 1) + j] + gap) {
    //         aligned_seq1 = seq1[i - 1] + aligned_seq1;
    //         aligned_seq2 = "-" + aligned_seq2;
    //         --i;
    //     } else {
    //         aligned_seq1 = "-" + aligned_seq1;
    //         aligned_seq2 = seq2[j - 1] + aligned_seq2;
    //         --j;
    //     }
    // }

    // while (i > 0) {
    //     aligned_seq1 = seq1[i - 1] + aligned_seq1;
    //     aligned_seq2 = "-" + aligned_seq2;
    //     --i;
    // }

    // while (j > 0) {
    //     aligned_seq1 = "-" + aligned_seq1;
    //     aligned_seq2 = seq2[j - 1] + aligned_seq2;
    //     --j;
    // }

    // std::cout << "Aligned Sequence 1: " << aligned_seq1 << std::endl;
    // std::cout << "Aligned Sequence 2: " << aligned_seq2 << std::endl;

    delete[] h_score;
    delete[] h_score_r;
}
