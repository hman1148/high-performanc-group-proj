# README

---

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

`salloc -n 1 -N 1 -t 0:15:00 -p notchpeak-gpu -A notchpeak-gpu --gres=GPU`

---

### Loading Modules

Use the following commands to load the `cuda` and `mpi` modules. These are necessary to build the project.

`module load cuda`


`module load mpi`

---

### Building the Executable

We use CMake to compile this project. From the `root` directory, navigate to the `build` directory.

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

`./class_project <k> <max_iterations> <tolerance> <algorithm_id>`

- `k`: This is setting the `k` value, or number of centroids, for the KMeans Algorithm. 
It should be an integer greater than or equal to 1.
- `max_iterations`: This is setting the maximum number of iterations that the KMeans Algorithm 
will perform if convergence is not reached earlier.It should be a positive integer greater than or equal to 1.
- `tolerance`: This variable sets the  tolerance minimum centroid movement required to continue iterations; used to determine convergence.
- `algorithm_id`: Selects which implementation to run for the K-Means algorithm.

    | ID | Implementation         |
    |----|------------------------|
    | 1  | Serial                 |
    | 2  | Shared Memory CPU      |
    | 3  | Distributed Memory CPU |
    | 4  | Shared Memory GPU      |
    | 5  | Distributed Memory GPU |

***Note***: Your first time running the program, you will notice that reading the `data/tracks_features.csv` file takes a few minutes. After the program has ran once, it will save the track features as a binary file, and reading will be much faster.

### Configuring the Problem Size

TODO: Make it so that after picking an implementation, the user can select what percentage of the track_features they would want to perform algoithm on

### Configuring Number of Threads (CPU)

TODO: Make it so that after picking an cpu implementation, the user is shown how many possible threads they can use and allow them to pick how many they can use

### Configuring Block Size (GPU)

Todo: After selecting a gpu implementation, the user can select the block size to use