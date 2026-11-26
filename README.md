# Hybrid OpenMP + MPI Box Blur

A C++17 image-processing project that applies a 2D box blur to all `.jpg`/`.jpeg` images in a directory using **MPI** (distributed across processes/hosts) and **OpenMP** (parallelized across threads per process), built with **OpenCV**.

## Folder Structure

```text
.
├── benchmark.sh        # Sweeps MPI process and OpenMP thread counts and times runs
├── depend.sh           # Installs system dependencies and initializes build/test folders
├── Makefile            # Builds build/main and runs it with mpirun + OpenMP
├── README.md           # Project documentation
├── build/              # Compiled objects, dependency file, and executable (build/main)
├── src/
│   └── main.cpp        # Hybrid MPI + OpenMP box blur implementation
├── template/           # Original reference template (program.c, Makefile, .gitignore)
└── test/
    ├── src/            # Input JPG images for testing
    └── out/            # Blurred output JPG images
```

## Dependencies & Setup

The project targets Ubuntu/Debian systems and requires:

- `build-essential` (`g++`, `make`) with OpenMP support
- `pkg-config` (used by `Makefile` to resolve `opencv4` compiler/linker flags)
- `openmpi-bin` and `libopenmpi-dev` (`mpiCC`, `mpirun`)
- `libopencv-dev` (OpenCV 4 `core` and `imgcodecs`)

Run the setup script to install all dependencies and initialize `build/` and `test/out/`:

```bash
./depend.sh
```

## How the Code Works

1. **Initialization & CLI Parsing (`src/main.cpp`)**:
   - Initializes MPI (`MPI_Init`, `MPI_Comm_size`, `MPI_Comm_rank`) and parses `-src <input_dir>` and `-out <output_dir>`.
2. **File Discovery & Synchronization**:
   - Rank `0` creates `<output_dir>`, scans `<input_dir>` for `.jpg`/`.jpeg` files (case-insensitive), sorts them, and broadcasts the file count (`MPI_Bcast`) so all ranks process images in lockstep.
3. **Image Distribution**:
   - For each image, rank `0` loads it via `cv::imread(..., cv::IMREAD_COLOR)`, broadcasts its dimensions and OpenCV type (`meta[3]`), and broadcasts the full raw pixel buffer (`MPI_Bcast`) so every rank has access to neighboring pixels up to kernel radius `R = 20` (a `41x41` box blur window).
4. **Hybrid MPI + OpenMP Row Partitioning**:
   - Total worker threads across the cluster is `total_threads = numprocs * omp_threads`.
   - Each thread computes its global thread ID (`global_tid = rank * nt + tid`) and uses `row_start()` to claim a balanced, contiguous slice of rows `[start_row, end_row)`.
   - All threads on a given MPI rank together cover one contiguous range `[rank_start, rank_end)`, stored in a rank-local buffer `local_out`.
   - Each pixel is averaged over its clamped `(2R+1)x(2R+1)` neighborhood across all channels.
5. **Gathering & Output**:
   - Rank `0` computes byte counts and displacements per rank and collects all slices with a single `MPI_Gatherv`, then writes the result to `<output_dir>/<filename>` via `cv::imwrite`.

## Building & Running

Build and run with the default settings (`MPI_NP=3`, `OMP_THREADS=4`, `-src test/src -out test/out`):

```bash
make
make run
```

You can override `Makefile` variables on the command line:

```bash
make run MPI_NP=2 OMP_THREADS=4 SRC_DIR=path/to/in OUT_DIR=path/to/out
```

Clean build artifacts:

```bash
make clean
```

## Benchmark

`benchmark.sh` builds the binary once with `make` and then sweeps every combination of `mpi_np` from `1` to `MPI_NP_MAX` and `omp_threads` from `1` to `OMP_THREADS_MAX`, printing a table of wall-clock execution times:

```bash
./benchmark.sh
```

Example output format:

```text
MPI_NP   OMP_THREADS  TOTAL_THREADS  TIME (s)
------   -----------  -------------  --------
1        1            1              ...
```

### Modifying the Benchmark

Edit the configuration variables at the top of `benchmark.sh`:

- **`MPI_NP_MAX`**: Maximum number of MPI processes to test (must not exceed the number of entries in `HOSTS`).
- **`OMP_THREADS_MAX`**: Maximum number of OpenMP threads per MPI process to test.
- **`HOSTS`**: Array of IP addresses/hostnames mapped by MPI rank (`rank 0 -> HOSTS[0]`, `rank 1 -> HOSTS[1]`, etc.). By default it uses `127.0.0.1` for local testing; replace entries with worker node IPs when benchmarking across multiple machines (passwordless SSH and matching project paths are required on remote hosts).
