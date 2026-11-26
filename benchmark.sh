#!/usr/bin/env bash
set -euo pipefail

# Define maximal values for parameter sweep
MPI_NP_MAX=3
OMP_THREADS_MAX=3

main() {
  cd "$(dirname "${BASH_SOURCE[0]}")"

  # Build the binary once before timing runs.
  make

  printf "\n%-8s %-12s %-14s %-10s\n" "MPI_NP" "OMP_THREADS" "TOTAL_THREADS" "TIME (s)"
  printf "%-8s %-12s %-14s %-10s\n" "------" "-----------" "-------------" "--------"

  # Sweep 1..MPI_NP and 1..OMP_THREADS and record elapsed wall-clock time.
  for ((mpi_np = 1; mpi_np <= MPI_NP_MAX; ++mpi_np)); do
    for ((omp_threads = 1; omp_threads <= OMP_THREADS_MAX; ++omp_threads)); do
      local start_ns end_ns elapsed_s total_threads
      total_threads=$((mpi_np * omp_threads))

      start_ns=$(date +%s%N)
      make -s run MPI_NP="$mpi_np" OMP_THREADS="$omp_threads" MPI_OPTIONS="--bind-to none --host localhost:${mpi_np}"
      end_ns=$(date +%s%N)

      elapsed_s=$(awk -v s="$start_ns" -v e="$end_ns" 'BEGIN { printf "%.4f", (e - s) / 1e9 }')
      printf "%-8d %-12d %-14d %-10s\n" "$mpi_np" "$omp_threads" "$total_threads" "$elapsed_s"
    done
  done
}

main "$@"
