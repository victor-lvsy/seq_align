#include "nw3.cuh"

// Test of a coarsening approach for the initialisation phase of the score matrix

// Kernel for initializing borders of the score matrix
__global__ void init_borders_v3(int *d_score, int n, int m, int gap, int coarsening_factor)
{
    for(int i=0; i<coarsening_factor;i++){
        int idx = blockIdx.x * blockDim.x + threadIdx.x + i;
        // Initialize first column (i.e., vertical border)
        if (idx <= n)
        {
            d_score[idx * (m + 1)] = idx * gap; // Setting gap penalties for the first column
        }

        // Initialize first row (i.e., horizontal border)
        if (idx <= m)
        {
            d_score[idx] = idx * gap; // Setting gap penalties for the first row
        }
    }

    
}

// Kernel for filling the matrix using the anti-diagonal and tiling approach
__global__ void fill_matrix_v3(int *d_score, const char *d_seq1, const char *d_seq2, int match, int mismatch, int gap, int n, int m, int diag_id, int offs) {

    int tid = threadIdx.x;   // Thread x-index within block
    int bid = blockIdx.x;
    int bdim = blockDim.x;

    int offsj = bid + offs;
    int offsi = diag_id - offsj;

    for (int diag = 2; diag <= 2 * bdim + 1; ++diag) {
        int first_i = max(1, diag - bdim); // Calculate start i index for the current diagonal
        int last_i = min(bdim, diag - 1);  // Calculate end i index for the current diagonal
        int elements_in_diag = last_i - first_i + 1; // Number of elements in the current diagonal

        if (tid < elements_in_diag) {
            int i = last_i - tid; // Mapping tid to i
            int j = diag - i + (offsj * bdim); // Mapping i to j
            i += offsi * bdim;
            int idx = i * (m + 1) + j;
            if (i <= n && j <= m) {
                int match_score = (d_seq1[i - 1] == d_seq2[j - 1]) ? match : mismatch;
                d_score[idx] = max(
                    d_score[(i - 1) * (m + 1) + (j - 1)] + match_score,
                    max(
                        d_score[(i - 1) * (m + 1) + j] + gap,
                        d_score[i * (m + 1) + (j - 1)] + gap));
            }
          }
        __syncthreads(); // Synchronize threads before moving to the next diagonal
}
}

// Host function for Needleman-Wunsch algorithm
void nw3(const std::string &seq1, const std::string &seq2, int match, int mismatch, int gap)
{
    printf("HELLO THERE, I am nw3\n");

    int n = seq1.length();
    int m = seq2.length();

    printf("Size of seq1: %d\nSize of seq2: %d\nProduct: %d\n", n, m, n * m);

    // Allocate the matrix with padding (extra row and column) on the host
    int *h_score = new int[(n + 1) * (m + 1)];
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

    // Initialize borders of the matrix on GPU
    int num_threads = max(n, m);
    int coarsening_factor = 16;
    int num_blocks = (num_threads + (NUMBER_OF_THREADS - 1)) / NUMBER_OF_THREADS;
    num_blocks = (num_blocks + (coarsening_factor - 1)) / coarsening_factor;
    printf("Number of blocks: %d\nNumber of Threads: %d\n", num_blocks, num_threads);
    init_borders_v3<<<num_blocks, NUMBER_OF_THREADS>>>(d_score, n, m, gap, coarsening_factor);
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());

    // Define tile size
    int grid = (n + TILE_SIZE - 1) / TILE_SIZE;

    printf("Number of blocks: {%d}\nNumber of Threads: %d\n", grid, TILE_SIZE);
    // Fill the matrix on GPU using tiled approach
    int offs = 1;
    for(int i=1; i<=2*grid+1; i++){
        if(i<=grid + 1){
            // printf("i: %d, grid size: %d\n", i, i);
            fill_matrix_v3<<<i, TILE_SIZE>>>(d_score, d_seq1, d_seq2, match, mismatch, gap, n, m, i - 1, 0);
            CHECK_KERNELCALL();
            CHECK(cudaDeviceSynchronize());
        }
        else{
            // printf("i: %d, grid size: %d, j: %d\n", i, ((2 * grid + 1 - i) + 1), offs);
            fill_matrix_v3<<<((2 * grid + 1 - i) + 1), TILE_SIZE>>>(d_score, d_seq1, d_seq2, match, mismatch, gap, n, m, i - 1, offs);
            CHECK_KERNELCALL();
            CHECK(cudaDeviceSynchronize());
            offs++;
        }
        
    }

    // Copy score matrix back to CPU from GPU
    CHECK(cudaMemcpy(h_score, d_score, (n + 1) * (m + 1) * sizeof(int), cudaMemcpyDeviceToHost));

    // Clean up GPU memory
    CHECK(cudaFree(d_score));
    CHECK(cudaFree(d_seq1));
    CHECK(cudaFree(d_seq2));

    // Print score matrix for debugging
    // for (int i = 0; i <= n; ++i) {
    //     for (int j = 0; j <= m; ++j) {
    //         std::cout << h_score[i * (m + 1) + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // Backtracking
    // int i = n;
    // int j = m;
    // std::string aligned_seq1, aligned_seq2;

    // while (i > 0 && j > 0) {
    //     if (h_score[i * (m + 1) + j] == h_score[(i - 1) * (m + 1) + (j - 1)] + (seq1[i - 1] == seq2[j - 1] ? match : mismatch)) {
    //         aligned_seq1 = seq1[i - 1] + aligned_seq1;
    //         aligned_seq2 = seq2[j - 1] + aligned_seq2;
    //         --i;
    //         --j;
    //     } else if (h_score[i * (m + 1) + j] == h_score[(i - 1) * (m + 1) + j] + gap) {
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
}
