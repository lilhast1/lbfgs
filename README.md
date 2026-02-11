# CUDA L-BFGS Optimization

A CUDA implementation of the limited-memory Broyden–Fletcher–Goldfarb–Shanno (L-BFGS) algorithm for large-scale unconstrained optimization. The solver uses mixed-precision arithmetic and custom GPU kernels for vector operations, with emphasis on optimizing dot-product reductions in the two-loop recursion.

---

## Features
- Mixed-precision arithmetic (float32 for runtime variables, float64 for reductions).
- Custom CUDA kernels:
  - `dot_partial_f32_to_f64`, `dot_atomic_f32`, `dotProduct`
  - `axpy`, `mulVecScal`, `setVectorScalar`
- GPU-based line search with fallback strategies.
- Benchmarks on Quadratic, Rosenbrock, Rastrigin, and Ackley functions.
- Comparative analysis vs. CPU baselines and cuBLAS (with/without line search).
- Scalability testing across dimensions up to 16M.

---

## Benchmark Highlights (N = 4096)

**Rosenbrock**
- CPU: 34,786 ms  
- CUDA L-BFGS: 157.9 ms (220× speedup, error 2.86e‑12)  
- cuBLAS: 31.6 ms (fails to converge, error ~1.50e+32)  
- cuBLAS+LS: 1153.6 ms (converges, error 9.59e‑13)

**Ackley**
- CPU: 1067 ms  
- CUDA L-BFGS: 16.6 ms (64× speedup)  
- cuBLAS: 45.4 ms (23× speedup)  
- cuBLAS+LS: 24.0 ms (44× speedup)

**Rastrigin**
- CPU: 1471 ms  
- CUDA L-BFGS: 17.5 ms (84× speedup)  
- cuBLAS+LS: 16.5 ms (89× speedup)

**Quadratic**
- CUDA L-BFGS: 85.4 ms (error 2.47e‑13)  
- cuBLAS: 14.0 ms (error 0)

---

## Requirements
- NVIDIA GPU with CUDA support (tested on Turing architecture).
- CUDA Toolkit 13.0+.
- C++17 compiler.
- Nsight Compute (optional, for profiling).

---

## Build
```bash
git clone https://github.com/lilhast1/lbfgs.git
cd lbfgs
nvcc -O3 lbfgs_mixed_precision.cu -o lbfgs
```

---

## Run
```bash
./lbfgs
```

---

## Future Work
- Extend testing to more benchmark functions.
- Explore multi-GPU scaling.
- Apply solver to real-world tasks (ML training, inverse problems, scientific simulation).

---

## Citation
If you use this code in your research: <br>
```bash
@article{lbfgs_cuda, 
  title={Mixed-Precision L-BFGS on CUDA: A Comparative Benchmark},
  author={Tarik Hastor and Ismar Muslić and Merjem Gutošić and Ivona Jozić and Kanita Kadušić}, 
  year={2026} 
}
```

---

## Authors

- [**Tarik Hastor**](https://github.com/lilhast1)
- [**Ismar Muslić**](https://github.com/imuslic1)  
- [**Merjem Gutošić**](https://github.com/mgutosic1)  
- [**Ivona Jozić**](https://github.com/ijozic1) 
- [**Kanita Kadušić**](https://github.com/kkadusic2)

**Faculty of Electrical Engineering, University of Sarajevo**

**Contact:**  
{thastor1, imuslic1, mgutosic1, ijozic1, kkadusic2}@etf.unsa.ba


