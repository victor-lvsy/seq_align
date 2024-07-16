# DNA Sequence Alignment in CUDA

## Project Overview
This project is part of the GPU and Heterogeneous Systems course at Politecnico di Milano. The goal is to implement and compare various approaches for DNA sequence alignment using CUDA.
This project is based on the https://en.wikipedia.org/wiki/Needleman%E2%80%93Wunsch_algorithm wikipedia page.
## Author
**Victor LEVESY**

---

## Test Files
There are multiple test files provided to evaluate the performance and accuracy of the implementations:

- **Short Files for Debugging**:
  - `test1.txt`
  - `test2.txt`
  - `test3.txt`
  - `test4.txt`

- **Longer Files for Performance Testing**:
  - `s1.txt`
  - `s2.txt`

---

## Implementations
This repository includes seven different implementations of the Needleman-Wunsch algorithm for DNA sequence alignment:

1. **nw.cpp**:
   - **Description**: A sequential approach running solely on the CPU.
   - **Purpose**: Serve as a baseline for performance comparison.

2. **nw1.cu**:
   - **Description**: A basic parallel approach using grid synchronization for longer outputs.
   - **Purpose**: Introduce parallelism and compare efficiencies with the sequential approach.

3. **nw2.cu**:
   - **Description**: A tiling approach that enables synchronization between blocks without requiring `grid.sync()`. This method necessitates launching additional kernels.
   - **Purpose**: Improve efficiency by reducing synchronization overhead.

4. **nw3.cu**:
   - **Description**: An attempt at thread coarsening during the initialization phase, though the results are not significantly improved.
   - **Purpose**: Experiment with thread coarsening as an optimization technique.

5. **nw4.cu**:
   - **Description**: Utilizes anti-diagonal memory management for coalesced memory accesses.
   - **Purpose**: Enhance memory access patterns to increase performance.
   - **Efficiency loss**: Can't process sequences longer than the number of thread in a block.

6. **nw5.cu**:
   - **Description**: Adds shared memory usage to store DNA sequences within blocks (on-chip storage).
   - **Purpose**: Explore the benefits of shared memory for frequently accessed data.
   - **Efficiency loss**: Can't process sequences longer than the number of thread in a block.

7. **nw6.cu**:
   - **Description**: Implements dynamic kernel launches to maximize GPU utilization. 
   - **Purpose**: Maximize GPU usage and explore dynamic kernel launch capabilities.
   - **Efficiency loss**: This is a simplified version of the algorithm, as longer sequences are split and processed independently.


