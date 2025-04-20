# README

---

## Project Layout

The Scaling Study and GPU implementation analysis can be found in the `ScalingStudy.md` file. The description of our implementations is in `implementations.md`. Our teamwork report is in `Teamwork.md`. Instructions for loading modules, building the project, running the project, etc. can be found below.

## Instructions

After cloning the repo, you will have to create a `data` folder in the root directory of the project. 
Once you have created the `data` folder, download the `tracks_features.csv` file from the following link: 
[kaggle.com](https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs?resource=download). 
Now insert the `tracks_features.csv` file into the `data` directory, and you are good to go.

Then ssh into CHPC. Your command will look like this: `ssh <uNUMBER>@notchpeak.chpc.utah.edu`

Now copy the project to you CHPC account. Your command will look something like this: 

`scp -r /local/path/high-performanc-group-proj <uNUMBER>@notchpeak.chpc.utah.edu:~/high-performanc-group-proj/data`

--- 

### Picking an Allocation

To view available allocations, use the command: `myalloc`.

We did our development on the `notchpeak-gpu` allocation. You can use other allocations, but keep in mind that some modules
may not be available on certain machines, and you will not be able to run the GPU implementations on a non GPU allocation.

To select your interactive allocation, your command will look something like this: 

`salloc -n 1 -N 1 -t 0:15:00 -p notchpeak-gpu -A notchpeak-gpu --gres=gpu`

---

### Loading Modules

Use the following commands to load the `cuda` and `mpi` modules. These are necessary to build the project.

`module load cuda`


`module load mpi`


### Loading Modules for Validation

If you wish to use the `validation.py` script to check the results of the output .csv files, then also load `pandas` with:

`module load pandas`

### Loading Module for Visualization

If you wish to visualize the clusters, then we recommend installing your own Python module. 

We used the following commands, but further instructions can be found on 
[chpc.utah.edu](https://www.chpc.utah.edu/documentation/software/python-anaconda.php)

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash ./Miniforge3-Linux-x86_64.sh -s -b -p $HOME/software/pkg/miniforge3

mkdir -p $HOME/MyModules/miniforge3
cp /uufs/chpc.utah.edu/sys/installdir/python/modules/miniforge3/latest.lua $HOME/MyModules/miniforge3

module use $HOME/MyModules
module load miniforge3/latest

mamba install pandas numpy matplotlib
```
After this, you should be good to go, but again refer to CHPCs documentation if you run into any issues.

---

### Building the Executable

We use `CMake` to compile this project. From the `root` directory, navigate to the `build` directory.

`cd build`

If for some reason the `build` directory does not exist, make it with:

`mkdir build`.

From inside the build directory, run:

`cmake ..`

`make`

Now you should have an executable named `class_project` inside your `build` directory.

--- 

### Running the Executable

There are five implementations of the KMeans algorithm available for you to run: 
1. Serial
2. Shared Memory Parallel CPU (OpenMP)
3. Distributed Memory Parallel CPU (MPI)
4. Shared Memory Parallel GPU (Cuda)
5. Distributed Memory Parallel GPU (Cuda)

To run any of the implementations, you will structure a command like the following 
(assuming you are still in the `build` directory):

`./class_project <k> <max_iterations> <tolerance> <dataset_percentage> <algorithm_id>`

- `k`: This is setting the `k` value, or number of centroids, for the KMeans Algorithm. 
It should be an integer greater than or equal to 1.
- `max_iterations`: This is setting the maximum number of iterations that the KMeans Algorithm 
will perform if convergence is not reached earlier.It should be a positive integer greater than or equal to 1.
- `tolerance`: This variable sets the  tolerance minimum centroid movement required to continue iterations; used to determine convergence.
- `dataset_percentage`: This argument determines the size of the dataset. Think of it as a percentage, using numbers 1 to 100. This will determine what percentage of the dataset you use for computation.
- `algorithm_id`: Selects which implementation to run for the K-Means algorithm.

    | ID | Implementation         |
    |----|------------------------|
    | 1  | Serial                 |
    | 2  | Shared Memory CPU      |
    | 3  | Distributed Memory CPU |
    | 4  | Shared Memory GPU      |
    | 5  | Distributed Memory GPU |


Here is an example command for running the Serial implementation:

`./class_project 3 100 0.0001 100 1`

***Note***: Your first time running the program, you will notice that reading the `data/tracks_features.csv` file takes a few minutes. After the program has ran once, it will save the track features as a binary file, and reading will be much faster.

### Configuring Number of Threads/Ranks (CPU)

If you wish to run the Shared Memory (OpenMP) implementation with a certain number of threads, use this command:

`export OPENMP_NUM_THREADS=<number of threads>`

To run the Distributed Memory CPU (MPI) implementation with a certain number of ranks, use a command like:

`mpirun n -<number of ranks> ./class_project 3 100 0.001 100 3`

--- 

### Running the Validation Script

Follow the steps above to load the correct modules. Then navigate to the `root` directory of the project.
From there, run:

`python3 validate.py <file1.csv> <file2.csv> [file3.csv ...]`.

For example, to compare the shared memory and the parallel memory CPU implementations, the command becomes:

`python3 validate.py serial_results.csv shared_cpu_results.csv`


--- 

### Running the Visualization Script

Follow the steps above to load the correct modules. Then navigate to the `root` directory of the project. From there, run:

`python3 visualize_clusters.py <file.csv>`

For example, to visualize the output of the serial implementation, the command becomes:

`python3 visualize_clusters.py serial_results.csv`

***Note***: *To keep execution time low, visualization is not run on the output. The user must run the above commands to generate a visualization*

