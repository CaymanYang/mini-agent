# Project conventions

This repository contains the mini local coding agent plus task workspaces under
`./WorkDir`.

## Layout

- `agent.py`: the mini ReAct agent.
- `README.md`: user-facing usage, backend, and benchmark documentation.
- `WorkDir/HipKernel`: HIP kernel experiments.
- `WorkDir/LeetCode`: algorithm practice files and tests.

## Running the mini-agent

- Prefer `python3 agent.py` on Linux and `python agent.py` on Windows.
- Use `--backend vulkan` or `--backend hip` when the requested llama.cpp backend
  matters. `--backend auto` is the default.
- Use `--server-bin` and `--model-path` only when auto-detection is not enough.

## HIP kernel workspace

The HIP project lives in `./WorkDir/HipKernel`, which has a `Makefile` with
these targets: `compile`, `run`, `clean`, `all` (= clean + compile + run).

- Build with `make -C ./WorkDir/HipKernel compile CASE=<name>`.
- Run with `make -C ./WorkDir/HipKernel run CASE=<name>`.
- Clean, build, and run with `make -C ./WorkDir/HipKernel all CASE=<name>`.
- The default `CASE` is `gemm`, so it builds `gemm.hip` if no case is given.
- To work on `rmsnorm.hip`, use `CASE=rmsnorm`.
- Do not invoke generated `.exe` files directly; run through the Makefile.
- Generated binaries such as `*.exe` are build artifacts and should not be
  committed.

This machine's GPU is `gfx1150`. HIP must be compiled with the matching arch
flag: `hipcc --offload-arch=gfx1150 ...` (the Makefile already does this).

## LeetCode workspace

Algorithm practice lives in `./WorkDir/LeetCode`.

- Python solutions can be run directly, for example
  `python3 ./WorkDir/LeetCode/problem1.py`.
- Existing Python tests can be run with
  `python3 ./WorkDir/LeetCode/test_problem1.py`.
- C++ solutions can be compiled to a temporary binary under `/tmp` or the same
  directory, but do not commit generated binaries.

## Paths

- Prefer forward slashes in commands and tool arguments, for example
  `./WorkDir/HipKernel/rmsnorm.hip`.
- Avoid relying on Windows-only paths unless the task is explicitly about
  Windows.

## Coding discipline

- Think first: state assumptions; if unclear, ask instead of guessing.
- Keep changes minimal and focused on the requested task.
- Prefer existing local patterns over new abstractions.
- Verify the change with the smallest relevant build, run, or test command.
- Do not modify generated files, logs, caches, or binaries unless explicitly
  requested.


