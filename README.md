# MC20_PRJ
SNU Muticore Computing 2020 Spring Final Term Project

## A : CPUs in a single node
1. OpenMP parallelization employed.

2. Loop orders changed in convolution and transposed convolution for improved memory access pattern.

3. Tiling(blocking) for the 2 convolutions applied.

4. A bunch of image can be processed at a time. 

Performance : 4.0 img/sec (64 images)

## B : A single GPU with CPUs in a single node
1. Padding for explicit vectorization in the various weight parameters and input/output buffer in each stage

2. Loop orders changed (like in A).

3. Shared memory used for normalization.

4. Elimination of branch divergence.

Performance : 49.8 img/sec (128 images)

## C : 4 GPUs with CPUS in a single node

1. B + MPI Parallelization

Performance : 159.4 img/sec (256 images)

## D : 16 GPUs with CPUs in 4 nodes

1. B + MPI Parallelization

Performance : 258.5 img/sec (512 images)
