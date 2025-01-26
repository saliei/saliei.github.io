---
title: "Tiled Matrix Multiplication"
date: "2025-01-25"
author: "Saeid"
description: "Tiled algorithms, in general, divide the problem into smaller, \
	manageable tiles that fit into faster, but limited-size memory, \ 
	be it cache, shared memory, or registers, to improve memory access patterns. \
	Many problems which often involve matrices, can be broken down into tiles. \
	This improves **cache** utilization which improves cache hit/miss ratio, \
	by reusing data within smaller subproblems. In this post we implement \
	tiled matrix multiplication."
---
The idea is simple, reuse the data already loaded into the cache as much as possible. 

Tiled algorithms, in general, divide the problem into smaller, manageable tiles that fit into faster, 
but limited-size memory, be it cache, shared memory, or registers, to improve memory access patterns.
Some examples of tiled algorithms include:

- Matrix Multiplication: Divides the input matrices into submatrices (tiles), 
and computes partial multiplications for each tile. 
It uses shared memory in GPUs to store tiles, reducing global memory accesses.

- Convolutions in Deep Learning: In 2D convolution, sliding windows (tiles) are applied 
to an input image matrix to compute feature maps and FFT-based convolutions to optimize performance.

- Stencil Computations: Stencil algorithms often involve updating the value of a grid cell 
based on its neighbors, e.g. in finite difference methods and heat diffusion simulations. 
Tiling breaks the grid into sub-grids, ensuring that neighboring cells are available in shared memory.

- Sorting Algorithms: Tiled merge and quick sort involve breaking the array into smaller chunks (tiles), 
making it possible to perform smaller sorts. 
These can be done in parallel or distributed with optimizations, e.g., SIMD vector optimizations.

- Fast Fourier Transform: FFT algorithms like the Cooley-Tukey algorithm are inherently tiled, 
they recursively break down the input signals into smaller parts for sub-transforms.

- N-Body Simulations: In n-body cosmological simulations or molecular dynamic simulations, 
we can break particles into spatial tiles (e.g. using the Barnes-Hut tree) 
and compute interactions only between nearby tiles.
 
These were just some limited examples. Many other problems which often involve matrices, 
can be broken down into tiles. This improves **cache** utilization which improves cache hit/miss ratio, 
by reusing data within smaller subproblems. Accessing shared memory or L1 cache takes ~ 1-5 cycles, 
L2 cache, ~10-20 cycles, accessing global memory or DRAM takes ~200-500 cycles. 
If we can use shared memory, we can improve data fetching latency by orders of magnitude. 
Also, these tiles can often be processed independently, making **parallelism** possible. 

In particular, here we are interested in tiled matrix multiplication. 
To understand it, we take a simple case of multiplying two \$4 \times 4\$ matrices. 
We divide each matrix into four \$2 \times 2\$ submatrices, so for matrix \$A\$ we would have:

$$
A = 
\begin{bmatrix}
    a_{00} & a_{01} & \vert & a_{02} & a_{03} \\
    a_{10} & a_{11} & \vert & a_{12} & a_{13} \\
    \hline
    a_{20} & a_{21} & \vert & a_{22} & a_{23} \\
    a_{30} & a_{31} & \vert & a_{32} & a_{33}
\end{bmatrix} = 
\begin{bmatrix}
	A_{00} & A_{01} \\
	A_{10} & A_{11}
\end{bmatrix}
$$

Then we can simply show that, \$ C_{ij} = A_{ik}B_{kj} \$, or:

$$
\begin{align}
\begin{bmatrix}
	C_{00} & C_{01} \\
	C_{10} & C_{11}
\end{bmatrix}  = 
\begin{bmatrix}
	A_{00}B_{00} + A_{01}B_{10} & A_{00}B_{01} + A_{01}B_{11} \\
	A_{10}B_{00} + A_{11}B_{10} & A_{10}B_{01} + A_{11}B_{11}
\end{bmatrix}
\end{align}
$$

Such that for example for \$C_{00}\$ we have:

$$
\begin{align}
c_{00} & = a_{00}b_{00} + a_{01}b_{10} + a_{02}b_{20} + a_{03}b_{30} \\
c_{01} & = a_{00}b_{01} + a_{01}b_{11} + a_{02}b_{21} + a_{03}b_{31} \\
c_{10} & = a_{10}b_{00} + a_{11}b_{10} + a_{12}b_{20} + a_{13}b_{30} \\
c_{11} & = a_{10}b_{01} + a_{11}b_{11} + a_{12}b_{21} + a_{13}b_{31} 
\end{align}
$$

So to compute \$C_{00}\$ we do two multiplication, \$A_{00}B_{00}\$, and \$A_{01}B_{10}\$. 

To write the CUDA kernel for tiled matrix multiplication, we model the computation such that:
- Every thread is responsible for one element of the output matrix \$C\$, so thread(0, 0) computes \$c_{00}\$.
- Every thread block is responsible for one tile of the output matrix \$C\$, 
such that block(0, 0) will load \$A_{00}\$ and \$B_{00}\$ into shared memory, 
do the partial matrix multiplication, then load the next tiles, 
\$A_{01}\$ and \$B_{10}\$ into shared memory and add the partial multiplication result to the previous one. 

This approach makes it possible that: 
We use the shared memory. We can choose apriori the tile width, such that the tile fits into shared memory, 
as opposed to normal matrix multiplication where we can only do dynamic global memory allocations. 
Every element of matrices \$A\$ and \$B\$ is used twice by each thread, 
which cuts down the global memory access by two, compared to the normal access pattern. 
For example element \$a_{00}\$ is used by both thread(0, 0) and thread(0, 1).

Let's look at the CUDA kernel:

```cpp
__global__ void matrix_mul(double* A, double* B, double* C) {
    // shared memory can be thought of as an explicitly managed L1 cache,
    // that is private to each block, can be useful if we need to access 
    // data more than once, by one thread or by threads within a block,
    __shared__ double t_A[TILE_WIDTH * TILE_WIDTH];
    __shared__ double t_B[TILE_WIDTH * TILE_WIDTH];

    // each thread block takes one tile of C,
    // row and col are the indices of an element of C
    size_t bx = blockIdx.x; size_t by = blockIdx.y;
    size_t tx = threadIdx.x; size_t ty = threadIdx.y;
    size_t row = by * TILE_WIDTH + ty;
    size_t col = bx * TILE_WIDTH + tx;
    
    double sum = 0.0;
    // for a tile size of 2 for a matrix of size 4, possible 
    // amount of movements in either x or y direction is 2 tiles,
    // or N / TILE_WIDTH with possible padding
    for (int i = 0; i < (N + TILE_WIDTH - 1) / TILE_WIDTH; i++) {
        //Every thread in a block will load one element from 
        // global matrices to the shared tile matrices.
        // we're moving in the x direction for matrix A,
        // select row from A, row is constant for A for this tile
        if ((row < N) && ((i * TILE_WIDTH + tx) < N)) {
            t_A[ty * TILE_WIDTH + tx] = A[(N * row) + (i * TILE_WIDTH + tx)];
        }
        else {
            t_A[ty * TILE_WIDTH + tx] = 0.0;
        }

        // we're moving in the y direction for matrix B.
        // select col from B, col is constant for B for this tile
        if ((col < N) && ((i * TILE_WIDTH + ty) < N)) {
            t_B[ty * TILE_WIDTH + tx] = B[(i * TILE_WIDTH + ty) * N + col];
        }
        else {
            t_B[ty * TILE_WIDTH + tx] = 0.0;
        }
        
        // ensure threads/warps are done writing and reading
        __syncthreads();

        // partial partial multiplications
        for (int k = 0; k < TILE_WIDTH; k++) {
            sum += t_A[ty * TILE_WIDTH + k] * t_B[k * TILE_WIDTH + tx];
        }
        
        // ensure multiplication is finished, 
        // before overriding the tiles.
        __syncthreads();
    }

    if ((row < N) && (col < N)) {
        C[row * N + col] = sum;
    }
}
```

First, we define tiles of dimension `TILE_WIDTH * TILE_WIDTH` for each input matrix with `__shared__` identifier, 
as we mentioned shared memory can be thought of as a programmable L1 cache, which is shared by threads of a thread block. 
Then we calculate the tiles of the output matrix, where each thread block is responsible for one tile of the output matrix. 

Note the indexing when we load from matrix \$A\$, for each element of one tile of \$A\$, 
we move `N * row` to get to the element's row, `i * TILE_WIDTH` to get to the tile, 
and inside each tile, we move `threadIdx.x` times to get to the element, 
so in total `(N * row) + (i * TILE_WIDTH) + threadIdx.x`, 
which corresponds to the element `t_A[threadIdx.y][threadIdx.x]` of the tile, 
or element `threadIdx.y * TILE_WIDTH + threadIdx.x`. 
For loading columns from matrix \$B\$ to its corresponding tile in shared memory, 
we move `i * TILE_WIDTH + threadIdx.y` downward to get to each element's row, 
and we move `col` to the right to get each element of the column.

For launching the kernel we specify the grid and block dimension such that in each block we have 
`TILE_WIDTH * TILE_WIDTH` threads, `(N / TILE_WIDTH) * (N / TILE_WIDTH)` blocks with possible padding, 
also for convenience we do 2D indexing:

```cpp
// 2D block with TILE_WIDTH * TILE_WIDTH threads
dim3 dim_block(TILE_WIDTH, TILE_WIDTH);
// 2D grid with N/TILE_WIDTH * N/TILE_WIDTH blocks
dim3 dim_grid((N + TILE_WIDTH - 1) / TILE_WIDTH, 
			  (N + TILE_WIDTH - 1) / TILE_WIDTH);

matrix_mul<<<dim_grid, dim_block>>>(d_A, d_B, d_C);
```

The complete source code can be viewed at: [saliei/cuda-projects](https://gitlab.com/saliei/cuda-projects).


