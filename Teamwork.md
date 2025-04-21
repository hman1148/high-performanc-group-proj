# Teamwork Report

---

Below is a list of who completed which parts of the project. We feel that there was an even distribution of the workload. We collaborated well and had now issues working as a team.

---

## Joe

- Implemented the algorithm factory pattern
- Implemented much of the `main.cpp` file.
- Implemented Serial and Shared Memory CPU algorithms
- Created utility function for MinMax scaling the vectors
- Created utility function for calculating distance between a point and a centroid
- Implemented the `validation.py` and `visualization.py` scripts
- Ran the scaling study for the CPU implementations
- Wrote the `README.md`

---

## Hunter Peart

- Implemented `Global.cu` and `Global.cuh` algorithms
- Implemented `Shared.cu` and `Shared.cuh` algorithms
- Added CMakeList.txt program for GPU code and implemented additional compilation directives w/ Joe
- Worked on making libraries and versions compatible for GPU and CPU implementations on CHPC examples: MPICH, MPI, CUDA
- Created utility files such as `SpotifyFrameReader.cpp` and `SpotifyFrameReader.hpp` to inject .csv data from the Kaggle project
- Implemented `SpotifyFrame.h` object to represent the data coming in from the .csv data to be used throughout the program

---

## Andy

- Implemented `DistributedCpu.h` file using MPI for distributed memory parallelism  
- Wrote logic for scattering and gathering data using `MPI_Scatterv` and `MPI_Gatherv`  
- Implemented centroid synchronization and convergence logic using `MPI_Allreduce` and `MPI_Bcast`  
- Reconstructed flattened input/output vectors into structured `Point` objects for computation and result aggregation  
- Created and wrote the `implementations.md` file summarizing all five algorithm implementations with pseudocode and descriptions

