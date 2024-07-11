#include "nw2.cuh"

// Kernel for initializing borders of the score matrix with tiling
__global__ void init_borders(int *d_score, int n, int m, int gap)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

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

// Kernel for filling the matrix using the anti-diagonal and tiling approach
__global__ void fill_matrix(int *d_score, const char *d_seq1, const char *d_seq2, int match, int mismatch, int gap, int n, int m, int tile_size)
{
    // Declaration of shared memory to store parts of the score matrix
    extern __shared__ int shmem[];

    int tx = threadIdx.x; // Thread x-index within block
    int ty = threadIdx.y; // Thread y-index within block

    int bx = blockIdx.x; // Block x-index within grid
    int by = blockIdx.y; // Block y-index within grid

    // Calculate the total number of diagonals in the grid
    int num_diagonals = gridDim.x + gridDim.y - 1;

    // Loop over all diagonals one by one
    for (int d = 0; d < num_diagonals; ++d)
    {
        // Process only the blocks on the current diagonal
        if (bx + by == d)
        {
            int x_start = bx * tile_size + 1; // Starting x index for the current block
            int y_start = by * tile_size + 1; // Starting y index for the current block

            int x = x_start + tx; // Global x index for the thread
            int y = y_start + ty; // Global y index for the thread

            if (x <= n && y <= m)
            {
                int score_diag = (d_seq1[x - 1] == d_seq2[y - 1]) ? match : mismatch;
                int diag_idx = tx * tile_size + ty;

                // Load the necessary parts of the score matrix into shared memory
                shmem[diag_idx] = d_score[(x - 1) * (m + 1) + (y - 1)];                       // Top-left value
                shmem[diag_idx + tile_size * tile_size] = d_score[(x - 1) * (m + 1) + y];     // Top value
                shmem[diag_idx + 2 * tile_size * tile_size] = d_score[x * (m + 1) + (y - 1)]; // Left value

                __syncthreads(); // Synchronize threads to ensure shared memory is fully loaded

                // Compute the score for the current cell using values in shared memory
                d_score[x * (m + 1) + y] = max(
                    shmem[diag_idx] + score_diag,
                    max(
                        shmem[diag_idx + tile_size * tile_size] + gap,
                        shmem[diag_idx + 2 * tile_size * tile_size] + gap));
            }

            __syncthreads(); // Synchronize threads before moving to the next diagonal
        }
        __syncthreads(); // Synchronize threads after processing each diagonal
    }
}

// Host function for Needleman-Wunsch algorithm
void nw1(const std::string &seq1, const std::string &seq2, int match, int mismatch, int gap)
{
    printf("HELLO THERE, I am nw1\n");

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
    int num_blocks = (num_threads + (NUMBER_OF_THREADS - 1)) / NUMBER_OF_THREADS;
    printf("Number of blocks: %d\nNumber of Threads: %d\n", num_blocks, num_threads);
    init_borders<<<num_blocks, NUMBER_OF_THREADS>>>(d_score, n, m, gap);
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());

    // Define tile size
    int tile_size = 16; // Adjust depending on shared memory limitations and problem size
    dim3 dimBlock(tile_size, tile_size);
    int grid_x = (n + tile_size - 1) / tile_size;
    int grid_y = (m + tile_size - 1) / tile_size;
    dim3 dimGrid(grid_x, grid_y);

    // Fill the matrix on GPU using tiled approach
    size_t shared_memory_size = 3 * tile_size * tile_size * sizeof(int);
    void *args[] = {&d_score, &d_seq1, &d_seq2, &match, &mismatch, &gap, &n, &m, &tile_size};
    cudaLaunchCooperativeKernel((void *)fill_matrix, dimGrid, dimBlock, args, shared_memory_size);
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());

    // Copy score matrix back to CPU from GPU
    CHECK(cudaMemcpy(h_score, d_score, (n + 1) * (m + 1) * sizeof(int), cudaMemcpyDeviceToHost));

    // Clean up GPU memory
    CHECK(cudaFree(d_score));
    CHECK(cudaFree(d_seq1));
    CHECK(cudaFree(d_seq2));

    // Clean up CPU memory
    delete[] h_score;
}