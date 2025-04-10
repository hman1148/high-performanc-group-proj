# Scaling Study

***Serial vs Shared Memory CPU vs Distributed Memory CPU***

---

### 💪 Strong Scaling Study (Fixed Problem Size)

For this strong study, we used  `k = 8`, `tolerance=1e-4`, and used the full dataset. We also set `max_iterations=100`, though this threshold was never reached and has no bearing on the computation time.

| Threads / Ranks | Serial Time (s) | Shared Memory (OpenMP) Time (s) | Distributed Memory (MPI) Time (s) | OpenMP Speedup | MPI Speedup | Shared Memory  Efficiency (%) | Distributed memory Efficiency (%) |
|-----------------|-----------------|---------------------------------|-----------------------------------|----------------|-------------|-------------------------------|-----------------------------------|
| 1               | 180.496         | 174.307                         | 51.9082                           | 1.00           | 1.00        | 100.0                         | 100.0                             |
| 2               | 180.496         | 87.750                          |                                   |                |             |                               |                                   |
| 4               | 180.496         | 43.881                          |                                   |                |             |                               |                                   |
| 8               | 180.496         | 14.233                          |                                   |                |             |                               |                                   |
| 16              | 180.496         | 16.3302                         |                                   |                |             |                               |                                   |
| 32              | 180.496         | 18.3752                         |                                   |                |             |                               |                                   |


### 😩 Weak Scaling Study