# Project conventions

The HIP project lives in `./WorkDir`, which has a `Makefile` with these targets:
`compile`, `run`, `clean`, `all` (= clean + compile + run).

## Building and running

- To run the compiled program, ALWAYS use the Makefile, never invoke the `.exe`
  directly. Use `make -C ./WorkDir run` (do NOT call `./WorkDir/hip_gemm.exe`).
- To build, use `make -C ./WorkDir compile`.
- To clean, build, and run in one step, use `make -C ./WorkDir all`.

## HIP compilation

- This machine's GPU is gfx1150. HIP must be compiled with the matching arch flag:
  `hipcc --offload-arch=gfx1150 ...` (the Makefile already does this).

## Paths

- Prefer forward slashes in commands and tool arguments (e.g. `./WorkDir/...`) to
  avoid JSON-escaping issues with backslashes.

## Coding discipline (Karpathy)

- Think first: state assumptions; if unclear, ask instead of guessing.
- Simplest thing that works: no extra features, abstractions, or speculative code.
- Surgical edits: change only what the task needs; don't refactor unrelated code.
- Goal-driven: define how you'll verify success, then check it (run/build/test).


