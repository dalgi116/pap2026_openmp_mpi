#include "mpi/mpi.h"
#include <omp.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// Case-insensitive check so only JPG/JPEG inputs are processed.
static bool is_jpg(const fs::path &p) {
  std::string ext = p.extension().string();
  for (char &c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return ext == ".jpg" || ext == ".jpeg";
}

// Compute the starting row for global thread g so rows are balanced across all threads
// and remain contiguous per MPI rank for a single MPI_Gatherv.
static int row_start(int g, int rows, int total) {
  int base = rows / total;
  int rem = rows % total;
  return g * base + std::min(g, rem);
}

int main(int argc, char *argv[]) {
  // Initialize MPI and query process count and rank for distributed coordination.
  int numprocs = 1, rank = 0;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Parse -src and -out CLI arguments and exit cleanly if directories are invalid.
  std::string src_dir, out_dir;
  for (int i = 1; i + 1 < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-src") src_dir = argv[++i];
    else if (arg == "-out") out_dir = argv[++i];
  }

  if (src_dir.empty() || out_dir.empty() || !fs::is_directory(src_dir)) {
    if (rank == 0) {
      std::cerr << "Usage: " << argv[0] << " -src <input_dir> -out <output_dir>\n";
    }
    MPI_Finalize();
    return 1;
  }

  // Rank 0 collects and sorts JPG files and ensures the output folder exists,
  // then broadcasts the file count so all ranks loop in lockstep.
  std::vector<fs::path> files;
  int num_files = 0;
  if (rank == 0) {
    fs::create_directories(out_dir);
    for (const auto &entry : fs::directory_iterator(src_dir)) {
      if (entry.is_regular_file() && is_jpg(entry.path())) {
        files.push_back(entry.path());
      }
    }
    std::sort(files.begin(), files.end());
    num_files = static_cast<int>(files.size());
  }

  MPI_Bcast(&num_files, 1, MPI_INT, 0, MPI_COMM_WORLD);

  constexpr int R = 20;
  int omp_threads = omp_get_max_threads();
  int total_threads = numprocs * omp_threads;

  for (int i = 0; i < num_files; ++i) {
    cv::Mat src;
    int meta[3] = {0, 0, 0};

    // Rank 0 loads the image and ensures contiguous memory for raw MPI transfer.
    if (rank == 0) {
      src = cv::imread(files[i].string(), cv::IMREAD_COLOR);
      if (!src.empty()) {
        if (!src.isContinuous()) src = src.clone();
        meta[0] = src.rows;
        meta[1] = src.cols;
        meta[2] = src.type();
      }
    }

    // Broadcast image dimensions/type so worker ranks can allocate buffers or skip unreadable files.
    MPI_Bcast(meta, 3, MPI_INT, 0, MPI_COMM_WORLD);
    if (meta[0] == 0) continue;

    int rows = meta[0], cols = meta[1], type = meta[2];
    if (rank != 0) src = cv::Mat(rows, cols, type);

    // Broadcast full pixel data because each thread needs neighboring rows up to radius R.
    int row_bytes = cols * static_cast<int>(src.elemSize());
    MPI_Bcast(src.data, rows * row_bytes, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Allocate only this rank's output slice to avoid unnecessary memory allocation and gather traffic.
    int rank_start = row_start(rank * omp_threads, rows, total_threads);
    int rank_end = row_start((rank + 1) * omp_threads, rows, total_threads);
    cv::Mat local_out(rank_end - rank_start, cols, type);
    int channels = src.channels();

    // Split this rank's rows across OpenMP threads using global thread IDs for fine-grained parallelism.
#pragma omp parallel num_threads(omp_threads)
    {
      int tid = omp_get_thread_num();
      int nt = omp_get_num_threads();
      int global_tid = rank * nt + tid;
      int start_row = row_start(global_tid, rows, numprocs * nt);
      int end_row = row_start(global_tid + 1, rows, numprocs * nt);

      // Apply (2R+1)x(2R+1) box blur to assigned rows, clamping kernel bounds at image edges.
      for (int y = start_row; y < end_row; ++y) {
        int ky_min = std::max(0, y - R);
        int ky_max = std::min(rows - 1, y + R);
        uchar *dst_row = local_out.ptr<uchar>(y - rank_start);

        for (int x = 0; x < cols; ++x) {
          int kx_min = std::max(0, x - R);
          int kx_max = std::min(cols - 1, x + R);
          int count = (ky_max - ky_min + 1) * (kx_max - kx_min + 1);
          int sum[4] = {0, 0, 0, 0};

          for (int ky = ky_min; ky <= ky_max; ++ky) {
            const uchar *src_row = src.ptr<uchar>(ky);
            for (int kx = kx_min; kx <= kx_max; ++kx) {
              for (int c = 0; c < channels; ++c) {
                sum[c] += src_row[kx * channels + c];
              }
            }
          }
          for (int c = 0; c < channels; ++c) {
            dst_row[x * channels + c] = static_cast<uchar>(sum[c] / count);
          }
        }
      }
    }

    // Rank 0 computes byte counts/offsets and gathers all ranks' blurred slices into dst.
    cv::Mat dst;
    std::vector<int> recvcounts, displs;
    if (rank == 0) {
      dst = cv::Mat(rows, cols, type);
      recvcounts.resize(numprocs);
      displs.resize(numprocs);
      for (int r = 0; r < numprocs; ++r) {
        int rs = row_start(r * omp_threads, rows, total_threads);
        int re = row_start((r + 1) * omp_threads, rows, total_threads);
        recvcounts[r] = (re - rs) * row_bytes;
        displs[r] = rs * row_bytes;
      }
    }

    MPI_Gatherv(local_out.data, (rank_end - rank_start) * row_bytes, MPI_UNSIGNED_CHAR,
                rank == 0 ? dst.data : nullptr, recvcounts.data(), displs.data(),
                MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Rank 0 writes the assembled blurred image to the output directory.
    if (rank == 0) {
      fs::path out_path = fs::path(out_dir) / files[i].filename();
      if (!cv::imwrite(out_path.string(), dst)) {
        std::cerr << "Failed to write " << out_path << '\n';
      }
    }
  }

  // Cleanly shut down the MPI runtime before exiting.
  MPI_Finalize();
  return 0;
}
