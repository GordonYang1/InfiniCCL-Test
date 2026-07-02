# InfiniCCL-Test

InfiniCCL-Test is the external consumer-side test repository for InfiniCCL. It
validates InfiniCCL the same way downstream projects should use it: through the
installed public headers, the installed shared library, and the CMake package
target `InfiniCCL::infiniccl`.

This repository is intended for internal InfiniCCL development. It is not a
replacement for the main InfiniCCL README, build guide, or in-tree examples.

## Current Status

The repository is still early in its test-framework setup.

- `pure_mpi_tests/` contains the current MPI-backed test source.
- InfiniCCL is consumed with `find_package(InfiniCCL REQUIRED)`.
- `INFINICCL_INSTALL` is still accepted as a backward-compatible search hint,
  but new builds should prefer `CMAKE_PREFIX_PATH` or `InfiniCCL_ROOT`.
- CTest is planned but not wired yet.

## Planned Test Layout

The intended framework is organized by execution path:

```text
pure_mpi_tests/    Tests that exercise InfiniCCL through the MPI backend path.
pure_ccl_tests/    Tests reserved for pure InfiniCCL C API coverage.
mix_tests/         Mixed-path and integration tests across backend/device paths.
```

Each directory should eventually cover the same operator matrix. The current
target is 9 operators with 10 operation scenarios per operator. Prefer one test
file per operator so the matrix remains easy to scan and register with CTest.

When CTest support is added, it should become the common entry point for local
and CI verification:

```bash
ctest --test-dir build --output-on-failure
```

Until then, launch the generated test executables through `icclrun`.

## Prerequisites

- CMake 3.18 or newer
- A C++17-capable compiler
- MPI development headers and runtime
- An InfiniCCL source checkout that can be built and installed by `icclrun`
- CUDA Toolkit, MACA SDK, or other vendor/runtime dependencies required by the
  specific test suite being built

See the main InfiniCCL README for the full library build requirements.

## Build

Use `icclrun` from the InfiniCCL installation to build InfiniCCL and this
external test repository through the same launcher path used for development
verification.

First install and configure InfiniCCL following the main InfiniCCL README. After
the environment setup is sourced, `icclrun` and the required environment
variables, such as `INFINICCL_ROOT`, should already be available.

Then prepare a local `cluster.yaml`. The following file is only an example for a
single NVIDIA GPU on localhost; adjust paths, slots, device type, CMake flags,
and backend environment variables for the actual test environment.

```yaml
common_dir: "/path/to/InfiniCCL-Test"
common_user: "your_user"
install_dir: "/path/to/InfiniCCL-Test"

nodes:
  - ip: "localhost"
    type: "nvidia"
    slots: 1
    cmake_flags: "-DUSE_CUDA=ON"
    backend_env:
      CUDA_VISIBLE_DEVICES: "0"
```

Build InfiniCCL, install it under the configured `install_dir`, build the test
target, and launch it:

```bash
icclrun --config cluster.yaml --build pure_mpi_tests/all_reduce
```

The executable path is relative to `build/<arch>/`, so use
`pure_mpi_tests/all_reduce` rather than only `all_reduce`.

For non-local or heterogeneous runs, adjust `nodes`, `slots`, `cmake_flags`, and
backend environment variables in `cluster.yaml` instead of invoking MPI
directly.

The tests should not manually add InfiniCCL include directories, link directly
to `libinfiniccl.so`, or set an InfiniCCL RPATH. Those properties come from the
imported target `InfiniCCL::infiniccl`.

## Run

After the first build, run the same executable through `icclrun` without
`--build`:

```bash
icclrun --config cluster.yaml pure_mpi_tests/all_reduce
```

Do not launch these tests with `mpirun` directly from this repository. `icclrun`
generates the wrapper, hostfile, architecture-specific build path, and runtime
environment expected by InfiniCCL.

## Adding Tests

Use these conventions for new tests:

- Put each test in the directory that matches the path being validated:
  `pure_mpi_tests/`, `pure_ccl_tests/`, or `mix_tests/`.
- Use one source file per operator, named with the operator's snake_case name.
- Keep test names aligned with the InfiniCCL public C API and operator naming.
- Include InfiniCCL through public headers such as
  `#include <infiniccl/infiniccl.h>`.
- Link through `InfiniCCL::infiniccl`; do not vendor generated headers or commit
  local copies of `include/infiniccl.h`.
- When CTest registration is added, update the CMake test registration in the
  same change as the new test source.
- Do not commit build directories, local binaries, logs, cluster-local
  configuration, or installed InfiniCCL artifacts.

For reduction-type tests, be careful with dtype semantics. Movement operations
and arithmetic reductions do not validate data in the same way, especially for
half-precision types.

## Repository Layout

```text
CMakeLists.txt         Test build configuration and InfiniCCL package lookup.
include/               Shared helper headers for test programs.
pure_mpi_tests/        Current MPI-path test sources.
pure_ccl_tests/        Planned pure CCL-path test sources.
mix_tests/             Planned mixed-path test sources.
```
