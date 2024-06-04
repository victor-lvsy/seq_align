#include "nw1.cuh"

// Kernel for initializing borders of the score matrix
__global__ void init_borders(int *d_score, int n, int m, int gap) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx <= n) {
        d_score[idx * (m + 1)] = idx * gap; // Initialize first column
    }
    if (idx <= m) {
        d_score[idx] = idx * gap; // Initialize first row
    }
}

// Kernel for filling the matrix using the anti-diagonal approach
__global__ void fill_matrix(int *d_score, const char *d_seq1, const char *d_seq2, int match, int mismatch, int gap, int n, int m) {
    int total_diagonals = n + m - 1;
    int tid = threadIdx.x;
    
    for (int diag = 2; diag <= total_diagonals; ++diag) {
        int i = diag - tid;
        int j = tid;
        if (i >= 1 && i <= n && j >= 1 && j <= m) {
            int idx = i * (m + 1) + j;
            int match_score = (d_seq1[i - 1] == d_seq2[j - 1]) ? match : mismatch;
            d_score[idx] = max(
                d_score[(i - 1) * (m + 1) + (j - 1)] + match_score,
                max(
                    d_score[(i - 1) * (m + 1) + j] + gap,
                    d_score[i * (m + 1) + (j - 1)] + gap
                )
            );
        }
    }
}

void nw1(const std::string &seq1, const std::string &seq2, int match, int mismatch, int gap) {
    printf("HELLO THERE, i am nw1\n");
    int n = seq1.length();
    int m = seq2.length();

    // Allocating matrix with padding
    int *h_score = new int[(n + 1) * (m + 1)];
    std::fill(h_score, h_score + (n + 1) * (m + 1), 0);
    
    int *d_score;
    char *d_seq1, *d_seq2;
    
    // Allocate memory on GPU
    cudaMalloc(&d_score, (n + 1) * (m + 1) * sizeof(int));
    cudaMalloc(&d_seq1, n * sizeof(char));
    cudaMalloc(&d_seq2, m * sizeof(char));
    
    // Copy sequences to GPU
    cudaMemcpy(d_seq1, seq1.data(), n * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_seq2, seq2.data(), m * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_score, h_score, (n + 1) * (m + 1) * sizeof(int), cudaMemcpyHostToDevice);
    
    int num_threads = max(n, m) + 1;
    int num_blocks = (num_threads + (NUMBER_OF_THREADS - 1)) / NUMBER_OF_THREADS;
    
    // Initialize borders of the matrix on GPU
    init_borders<<<num_blocks, NUMBER_OF_THREADS>>>(d_score, n, m, gap);
    
    // Fill the matrix on GPU
    fill_matrix<<<num_blocks, NUMBER_OF_THREADS>>>(d_score, d_seq1, d_seq2, match, mismatch, gap, n, m);
    
    // Copy score matrix back to CPU
    cudaMemcpy(h_score, d_score, (n + 1) * (m + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Clean up GPU memory
    cudaFree(d_score);
    cudaFree(d_seq1);
    cudaFree(d_seq2);

    // Print score matrix for debugging
    // for (int i = 0; i <= n; ++i) {
    //     for (int j = 0; j <= m; ++j) {
    //         std::cout << h_score[i * (m + 1) + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // Backtracking
    int i = n;
    int j = m;
    std::string aligned_seq1, aligned_seq2;

    while (i > 0 && j > 0) {
        if (h_score[i * (m + 1) + j] == h_score[(i - 1) * (m + 1) + (j - 1)] + (seq1[i - 1] == seq2[j - 1] ? match : mismatch)) {
            aligned_seq1 = seq1[i - 1] + aligned_seq1;
            aligned_seq2 = seq2[j - 1] + aligned_seq2;
            --i;
            --j;
        } else if (h_score[i * (m + 1) + j] == h_score[(i - 1) * (m + 1) + j] + gap) {
            aligned_seq1 = seq1[i - 1] + aligned_seq1;
            aligned_seq2 = "-" + aligned_seq2;
            --i;
        } else {
            aligned_seq1 = "-" + aligned_seq1;
            aligned_seq2 = seq2[j - 1] + aligned_seq2;
            --j;
        }
    }

    while (i > 0) {
        aligned_seq1 = seq1[i - 1] + aligned_seq1;
        aligned_seq2 = "-" + aligned_seq2;
        --i;
    }

    while (j > 0) {
        aligned_seq1 = "-" + aligned_seq1;
        aligned_seq2 = seq2[j - 1] + aligned_seq2;
        --j;
    }

    std::cout << "Aligned Sequence 1: " << aligned_seq1 << std::endl;
    std::cout << "Aligned Sequence 2: " << aligned_seq2 << std::endl;

    delete[] h_score;
}
