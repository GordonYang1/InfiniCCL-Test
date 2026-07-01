#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include <infiniccl/infiniccl.h>

#include "runtime_api.h"

namespace {

constexpr int kSkipReturnCode = 77;

struct Options {
  std::string dtype = "float32";
  std::string red_op = "sum";
  std::string test_case = "basic";
  size_t count = 1024;
  bool pin_device0 = false;
};

void PrintUsage(const char *program) {
  std::cerr << "Usage: " << program
            << " --dtype <int32|float32|float64|float16|bfloat16>"
            << " --red-op <sum|prod|max|min|avg>"
            << " --count <N>"
            << " [--case <basic>]"
            << " [--pin-device0 <0|1>]" << std::endl;
}

bool ParseBool(const std::string &value, bool *out) {
  if (value == "1" || value == "true" || value == "on") {
    *out = true;
    return true;
  }
  if (value == "0" || value == "false" || value == "off") {
    *out = false;
    return true;
  }
  return false;
}

bool ParseSize(const std::string &value, size_t *out) {
  char *end = nullptr;
  unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
  if (end == value.c_str() || *end != '\0' || parsed == 0) {
    return false;
  }
  *out = static_cast<size_t>(parsed);
  return true;
}

bool ParseArgs(int argc, char **argv, Options *opts) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto require_value = [&](const char *name) -> const char * {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for `" << name << "`." << std::endl;
        return nullptr;
      }
      return argv[++i];
    };

    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else if (arg == "--dtype") {
      const char *value = require_value("--dtype");
      if (!value)
        return false;
      opts->dtype = value;
    } else if (arg == "--red-op") {
      const char *value = require_value("--red-op");
      if (!value)
        return false;
      opts->red_op = value;
    } else if (arg == "--count") {
      const char *value = require_value("--count");
      if (!value)
        return false;
      if (!ParseSize(value, &opts->count)) {
        std::cerr << "`--count` must be a positive integer." << std::endl;
        return false;
      }
    } else if (arg == "--case") {
      const char *value = require_value("--case");
      if (!value)
        return false;
      opts->test_case = value;
    } else if (arg == "--pin-device0") {
      const char *value = require_value("--pin-device0");
      if (!value)
        return false;
      if (!ParseBool(value, &opts->pin_device0)) {
        std::cerr << "`--pin-device0` must be 0/1, true/false, or on/off."
                  << std::endl;
        return false;
      }
    } else {
      std::cerr << "Unknown argument `" << arg << "`." << std::endl;
      return false;
    }
  }

  return true;
}

bool ParseRedOp(const std::string &name, infinicclRedOp_t *op) {
  if (name == "sum") {
    *op = infinicclSum;
  } else if (name == "prod") {
    *op = infinicclProd;
  } else if (name == "max") {
    *op = infinicclMax;
  } else if (name == "min") {
    *op = infinicclMin;
  } else if (name == "avg") {
    *op = infinicclAvg;
  } else {
    return false;
  }
  return true;
}

bool IsUnsupportedReductionDtype(const std::string &dtype) {
  return dtype == "float16" || dtype == "bfloat16";
}

int LocalRankFromEnv() {
  const char *env_names[] = {"OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID",
                             "MV2_COMM_WORLD_LOCAL_RANK"};
  for (const char *env_name : env_names) {
    const char *value = std::getenv(env_name);
    if (value != nullptr) {
      return std::atoi(value);
    }
  }
  return 0;
}

bool CheckInfini(infinicclResult_t result, const char *expr, int line) {
  if (result == infinicclSuccess) {
    return true;
  }
  std::cerr << "[InfiniCCL Error] `" << expr << "` returned " << result
            << " at line " << line << "." << std::endl;
  return false;
}

#define CHECK_INFINI_OR_RETURN(expr)                                           \
  do {                                                                         \
    if (!CheckInfini((expr), #expr, __LINE__)) {                               \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  } while (0)

double InputValueForRankAndIndex(int rank, int world_size, size_t index,
                                 infinicclRedOp_t op) {
  const double index_offset = static_cast<double>(index % 7);

  switch (op) {
  case infinicclProd:
    return rank == static_cast<int>(index % static_cast<size_t>(world_size))
               ? 2.0
               : 1.0;
  case infinicclAvg:
    return 2.0 * static_cast<double>(rank + 1) + 2.0 * index_offset;
  case infinicclSum:
  case infinicclMax:
  case infinicclMin:
    return static_cast<double>(rank + 1) + index_offset;
  default:
    return 0.0;
  }
}

template <typename T>
T InputValueForRankAndIndex(int rank, int world_size, size_t index,
                            infinicclRedOp_t op) {
  return static_cast<T>(InputValueForRankAndIndex(rank, world_size, index, op));
}

template <typename T>
T ExpectedValueForIndex(int world_size, size_t index, infinicclRedOp_t op) {
  double result = 0.0;

  switch (op) {
  case infinicclSum:
  case infinicclAvg:
    for (int rank = 0; rank < world_size; ++rank) {
      result += InputValueForRankAndIndex(rank, world_size, index, op);
    }
    if (op == infinicclAvg) {
      result /= static_cast<double>(world_size);
    }
    break;
  case infinicclProd:
    result = 1.0;
    for (int rank = 0; rank < world_size; ++rank) {
      result *= InputValueForRankAndIndex(rank, world_size, index, op);
    }
    break;
  case infinicclMax:
    result = InputValueForRankAndIndex(0, world_size, index, op);
    for (int rank = 1; rank < world_size; ++rank) {
      result = std::max(result,
                        InputValueForRankAndIndex(rank, world_size, index, op));
    }
    break;
  case infinicclMin:
    result = InputValueForRankAndIndex(0, world_size, index, op);
    for (int rank = 1; rank < world_size; ++rank) {
      result = std::min(result,
                        InputValueForRankAndIndex(rank, world_size, index, op));
    }
    break;
  default:
    result = 0.0;
    break;
  }

  return static_cast<T>(result);
}

template <typename T> bool EqualEnough(T actual, T expected) {
  if constexpr (std::is_integral_v<T>) {
    return actual == expected;
  } else {
    const double actual_value = static_cast<double>(actual);
    const double expected_value = static_cast<double>(expected);
    const double diff = std::fabs(actual_value - expected_value);
    const double scale = std::max(1.0, std::fabs(expected_value));
    return diff <= 1e-5 * scale;
  }
}

template <typename T>
bool ValidateAllReduce(const std::vector<T> &data, int rank, int world_size,
                       const Options &opts, infinicclRedOp_t red_op) {
  bool correct = true;
  size_t error_count = 0;
  T first_expected = ExpectedValueForIndex<T>(world_size, 0, red_op);

  for (size_t i = 0; i < data.size(); ++i) {
    T expected = ExpectedValueForIndex<T>(world_size, i, red_op);
    if (!EqualEnough(data[i], expected)) {
      correct = false;
      ++error_count;
      if (rank == 0 && error_count <= 3) {
        std::cerr << "Mismatch at index " << i << ": got "
                  << static_cast<double>(data[i]) << ", expected "
                  << static_cast<double>(expected) << "." << std::endl;
      }
    }
  }

  if (rank == 0) {
    const char *green = "\033[32m";
    const char *red = "\033[31m";
    const char *reset = "\033[0m";

    std::cout << "\n=== AllReduce Test Results ===" << std::endl;
    std::cout << "Case: dtype=" << opts.dtype << ", red-op=" << opts.red_op
              << ", count=" << opts.count << ", case=" << opts.test_case
              << std::endl;
    std::cout << "Correct: "
              << (correct ? (green + std::string("YES") + reset)
                          : (red + std::string("NO") + reset));
    if (!correct) {
      std::cout << " (" << error_count << " errors)";
    }
    std::cout << std::endl;
    if (!data.empty()) {
      std::cout << "Expect:  " << static_cast<double>(first_expected)
                << std::endl;
      std::cout << "Actual:  " << static_cast<double>(data[0]) << std::endl;
    }
  }

  return correct;
}

template <typename T>
int RunAllReduce(int argc, char **argv, const Options &opts,
                 infinicclDataType_t data_type, infinicclRedOp_t red_op) {
  if (opts.count > std::numeric_limits<size_t>::max() / sizeof(T)) {
    std::cerr << "Requested element count overflows byte size." << std::endl;
    return EXIT_FAILURE;
  }

  CHECK_INFINI_OR_RETURN(infinicclInit(&argc, &argv));

  int rank = 0;
  int world_size = 0;
  CHECK_INFINI_OR_RETURN(infinicclGetRank(&rank));
  CHECK_INFINI_OR_RETURN(infinicclGetSize(&world_size));
  if (world_size <= 0) {
    std::cerr << "Invalid MPI world size `" << world_size << "`." << std::endl;
    return EXIT_FAILURE;
  }

  char hostname[256];
  gethostname(hostname, sizeof(hostname));

  int device_id = opts.pin_device0 ? 0 : LocalRankFromEnv();
  CHECK_DEVICE(GPU_SET_DEVICE(device_id));
  CHECK_DEVICE(GPU_SYNC());

  gpuProp_t prop;
  CHECK_DEVICE(GPU_GET_DEVICE_PROPS(&prop, device_id));

  std::cout << "[Rank " << rank << "] Host: " << hostname
            << " | GPU: " << GPU_PLATFORM << " " << prop.name << " | Device "
            << device_id << std::endl;

  infinicclComm_t comm = nullptr;
  std::vector<int> device_ids;
  const int *device_list = nullptr;
  if (opts.pin_device0) {
    device_ids.assign(static_cast<size_t>(world_size), 0);
    device_list = device_ids.data();
  }
  CHECK_INFINI_OR_RETURN(infinicclCommInitAll(&comm, world_size, device_list));

  std::vector<T> h_send(opts.count);
  std::vector<T> h_recv(opts.count, static_cast<T>(0));
  for (size_t i = 0; i < opts.count; ++i) {
    h_send[i] = InputValueForRankAndIndex<T>(rank, world_size, i, red_op);
  }

  T *d_send = nullptr;
  T *d_recv = nullptr;
  size_t total_bytes = opts.count * sizeof(T);
  CHECK_DEVICE(GPU_MALLOC(&d_send, total_bytes));
  CHECK_DEVICE(GPU_MALLOC(&d_recv, total_bytes));
  CHECK_DEVICE(GPU_MEMCPY_H2D(d_send, h_send.data(), total_bytes));
  CHECK_DEVICE(GPU_MEMCPY_H2D(d_recv, h_recv.data(), total_bytes));
  CHECK_DEVICE(GPU_SYNC());

  CHECK_INFINI_OR_RETURN(infinicclAllReduce(d_send, d_recv, opts.count,
                                            data_type, red_op, comm, nullptr));
  CHECK_DEVICE(GPU_SYNC());
  CHECK_DEVICE(GPU_MEMCPY_D2H(h_recv.data(), d_recv, total_bytes));

  bool correct = ValidateAllReduce(h_recv, rank, world_size, opts, red_op);

  CHECK_DEVICE(GPU_FREE(d_send));
  CHECK_DEVICE(GPU_FREE(d_recv));
  CHECK_INFINI_OR_RETURN(infinicclCommDestroy(comm));
  CHECK_INFINI_OR_RETURN(infinicclFinalize());

  return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}

} // namespace

int main(int argc, char **argv) {
  Options opts;
  if (!ParseArgs(argc, argv, &opts)) {
    PrintUsage(argv[0]);
    return EXIT_FAILURE;
  }

  if (opts.test_case != "basic") {
    std::cerr << "Unsupported `--case` value `" << opts.test_case << "`."
              << std::endl;
    PrintUsage(argv[0]);
    return EXIT_FAILURE;
  }

  infinicclRedOp_t red_op = infinicclSum;
  if (!ParseRedOp(opts.red_op, &red_op)) {
    std::cerr << "Unsupported `--red-op` value `" << opts.red_op << "`."
              << std::endl;
    PrintUsage(argv[0]);
    return EXIT_FAILURE;
  }

  if (opts.dtype == "float32") {
    return RunAllReduce<float>(argc, argv, opts, infinicclFloat32, red_op);
  }
  if (opts.dtype == "float64") {
    return RunAllReduce<double>(argc, argv, opts, infinicclFloat64, red_op);
  }
  if (opts.dtype == "int32") {
    return RunAllReduce<int32_t>(argc, argv, opts, infinicclInt32, red_op);
  }
  if (IsUnsupportedReductionDtype(opts.dtype)) {
    std::cerr << "Skipping `" << opts.dtype
              << "` all_reduce reduction cases because the current MPI backend "
                 "maps fp16/bf16 to bytes instead of typed reduction values."
              << std::endl;
    return kSkipReturnCode;
  }

  std::cerr << "Unsupported `--dtype` value `" << opts.dtype << "`."
            << std::endl;
  PrintUsage(argv[0]);
  return EXIT_FAILURE;
}
