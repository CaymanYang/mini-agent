# mini local agent

A minimal ReAct coding agent: single file, zero third-party dependencies (Python standard library only), connecting to a local `llama-server` (OpenAI-compatible API).

Design goal: faster than Roo / Cline. Local models only do prompt processing at ~40-70 tok/s, and those tools inject tens of thousands of tokens of tool instructions every turn; this agent's system prompt is only a few hundred tokens and it reuses conversation history to hit llama.cpp's prompt cache.

## Prerequisites

1. Python 3.7+
2. A built `llama-server` and a `.gguf` model. The agent **launches `llama-server` for you** on startup (and shuts it down on exit), so you no longer need to start it manually. By default it auto-detects:
   - `llama-server` from the current directory, `LLAMA_CPP_DIR`, or a sibling `llama.cpp` checkout.
   - backend build directories such as `build-vulkan`, `build-x64-windows-vulkan-release`, or `build-hip`.
   - the model at `mymodels/gemma-4-12b-it-GGUF/gemma-4-12b-it-Q4_K_M.gguf`, falling back to the first `.gguf` under `mymodels`.

   Select a backend with `--backend vulkan` or `--backend hip`; the default `--backend auto` tries Vulkan first, then HIP. Override paths with `--server-bin` / `--model-path`. If a server is already running at the target URL, the agent detects it and reuses it instead of starting a new one. Use `--no-server` to skip launching entirely and connect to an existing server.

## llama.cpp backend requirements

You need a `llama-server` binary with the backend you want to use. You can use a packaged/prebuilt llama.cpp release if it includes the right backend, or build llama.cpp from source.

Platform support:

- Vulkan supports both Windows and Linux.
- ROCm/HIP is Linux-only for llama.cpp.

Common requirements:

- A `.gguf` model file, for example `gemma-4-12b-it-Q4_K_M.gguf`.
- CMake and a C/C++ compiler.
- Vulkan builds need a Vulkan-capable GPU driver and Vulkan SDK/tools (`glslc`).
- ROCm/HIP builds need ROCm installed and AMD GPU access. On Linux, the user usually needs to be in the `render` and `video` groups.

Build llama.cpp with Vulkan on Windows:

```powershell
cd llama.cpp
cmake -S . -B build-vulkan -DGGML_VULKAN=ON
cmake --build build-vulkan --config Release --target llama-server llama-bench
```

Build llama.cpp with Vulkan on Linux:

```bash
cd llama.cpp
cmake -S . -B build-vulkan -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan --target llama-server llama-bench -- -j"$(nproc)"
```

Build llama.cpp with ROCm/HIP on Linux:

```bash
cd llama.cpp
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
cmake -S . -B build-hip -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-hip --target llama-server llama-bench -- -j"$(nproc)"
```

After building, pass the matching server path with `--server-bin`, or update the default path in `agent.py`.

## Usage

```powershell
python agent.py
```

This starts `llama-server`, waits until its `/health` endpoint is ready, then drops you into the task prompt. The server's stdout/stderr is written to `mini-agent\server.log`. Enter a task, for example:

```
task> Read llama.cpp/src/models/gemma4.cpp and summarize its structure
```

Use `exit()` or `退出` to quit the agent program. Use `结束任务` or similar wording to end the current session, save compressed memory if enabled, and wait for a new task. Empty input is ignored.

## Options

| Option | Default | Description |
| --- | --- | --- |
| `--base-url` | built from `--host`/`--port` | OpenAI-compatible endpoint (overrides host/port) |
| `--model` | `local` | Model name (usually ignored by llama-server, any value works) |
| `--yes` | off | Skip confirmation before writing files / running commands |
| `--max-steps` | `20` | Max ReAct steps per task |
| `--max-tokens` | `2048` | Max tokens generated per step (prevents long unresponsive runs) |
| `--think` | off | Enable model reasoning/thinking; disabled by default to avoid stalls |
| `--work-dir` | `.` | Project root for tools, conventions, and `.mini-agent` memory |
| `--no-server` | off | Do not launch llama-server; connect to an already-running one |
| `--host` | `127.0.0.1` | llama-server host |
| `--port` | `8080` | llama-server port |
| `--backend` | `auto` | Backend build to auto-detect: `auto`, `vulkan`, or `hip` |
| `--server-bin` | auto-detect | Path to the `llama-server` executable |
| `--model-path` | auto-detect | Path to the `.gguf` model to serve |
| `--ngl` | `99` | Number of layers to offload to GPU |
| `--ctx-size` | `0` | Context size (`-c`); `0` leaves the server default |
| `--server-arg ARG` | none | Extra raw argument passed to llama-server (repeatable) |
| `--server-timeout` | `300` | Seconds to wait for the server to become ready |
| `--tokens` / `--no-tokens` | on | Show or hide input token estimates and per-step/session usage |
| `--auto-compact` / `--no-auto-compact` | on | Summarize old history when context usage reaches the threshold |
| `--compact-threshold` | `0.70` | Context usage ratio that triggers compaction |
| `--compact-keep-messages` | `6` | Recent messages kept verbatim after compaction |
| `--compact-max-tokens` | `1024` | Max tokens generated for the compaction summary |
| `--memory-file` | `.mini-agent/memory/session.md` | Markdown file used for compressed session memory |

Example (auto-confirm, limit to 10 steps):

```powershell
python agent.py --yes --max-steps 10
```

Example (choose a backend build):

```powershell
python agent.py --backend vulkan
```

```bash
python3 agent.py --backend hip
```

Example (run the agent from its source directory, but work on another project):

```bash
python3 /path/to/mini-agent/agent.py --work-dir /path/to/llama.cpp --backend hip
```

Example (serve a different model, pass extra server flags):

```powershell
python agent.py --model-path models\my.gguf --ctx-size 8192 --server-arg --flash-attn
```

Example (reuse a server you already started elsewhere):

```powershell
python agent.py --no-server --base-url http://127.0.0.1:8080/v1
```

## Project conventions (memory)

For persistent project rules (e.g. "always run the program via `make run`", "compile HIP with `--offload-arch=gfx1150`"), put them in an `AGENTS.md` file in the working directory. On startup the agent auto-loads it and injects it into the system prompt.

- Auto-detected filenames (first match wins): `AGENTS.md`, `AGENT.md`, `.agentrc`, `conventions.md`.
- Override the path with `--notes path\to\file.md`.

This is how you stop the agent from, e.g., invoking `./hip_gemm.exe` directly when you want `make run`.

## Long-memory compaction

The agent keeps the live chat history for prompt-cache reuse, but it now has a minimal long-memory path for longer tasks. When the estimated input context reaches `--compact-threshold` (default `0.70`) of `--ctx-size`, it asks the local model to summarize the older conversation, writes the summary to `.mini-agent/memory/session.md`, then rebuilds history as:

```text
system prompt + compressed session memory + last 6 messages
```

If `--ctx-size` is `0`, compaction uses an 8192-token fallback limit for the threshold check. Pass the actual server context with `--ctx-size` for better timing.

The memory file is loaded automatically on the next startup, so a later session can continue from the compressed task state. Disable this behavior with `--no-auto-compact`.

Memory is relative to `--work-dir`, not the agent source directory. For example, if you work on `llama.cpp` with `--work-dir /path/to/llama.cpp`, the task memory lives under `/path/to/llama.cpp/.mini-agent/`. The agent source directory can keep its own `.mini-agent` only for tasks about the agent itself.

## Tools

- `list_dir {path?}`: list directory contents (default: current directory), helps locate files via relative paths
- `read_file {path}`: read a file (with line numbers, truncated if too long)
- `write_file {path, content}`: create / overwrite a whole file (confirmation required by default)
- `str_replace {path, old_string, new_string}`: local replacement in an existing file (must match uniquely, confirmation required by default); prefer this for small edits to avoid rewriting the whole file
- `run_shell {command}`: run a command (PowerShell on Windows, confirmation required by default, 120s timeout)
- `finish {answer}`: end and give the final answer

## Protocol

The model emits one action per step:

```
ACTION
{"tool": "read_file", "args": {"path": "src/foo.py"}}
```

After execution the result is fed back to the model as an `OBSERVATION`, looping until `finish`.

## Safety notes

- `write_file` and `run_shell` print their content and ask for confirmation before running by default; use `--yes` to skip.
- Reads/outputs have size caps to avoid overly long content slowing down the model.

## Build and run (hip_gemm)

Use the fixed compile and run commands:

```powershell
hipcc --offload-arch=gfx1150 .\hip_gemm.hip -o hip_gemm.exe
```

Then run the generated program:

```powershell
.\hip_gemm.exe
```

## Local model performance (ROCm/HIP and Vulkan)

Backend model used by the agent: `gemma-4-12b-it-Q4_K_M.gguf` (6.62 GiB, 11.91 B params), measured with `llama-bench` on Linux.

ROCm/HIP device: AMD Radeon Graphics, `gfx1150` (0x1150), VMM: no, wave size: 32, VRAM: 30947 MiB.

Vulkan device: AMD Radeon Graphics (RADV GFX1150) (`radv`) | uma: 1 | fp16: 1 | bf16: 0 | warp size: 64 | shared memory: 65536 | int dot: 1 | matrix cores: KHR_coopmat.

| model | size | params | backend | ngl | test | t/s |
| --- | --: | --: | --- | --: | --- | --: |
| gemma4 ?B Q4_K - Medium | 6.62 GiB | 11.91 B | ROCm | 999 | pp128 | 260.66 ± 5.06 |
| gemma4 ?B Q4_K - Medium | 6.62 GiB | 11.91 B | ROCm | 999 | tg128 | 10.15 ± 0.01 |
| gemma4 ?B Q4_K - Medium | 6.62 GiB | 11.91 B | ROCm | 999 | tg256 | 10.11 ± 0.01 |
| gemma4 ?B Q4_K - Medium | 6.62 GiB | 11.91 B | Vulkan | 999 | pp128 | 198.52 ± 1.28 |
| gemma4 ?B Q4_K - Medium | 6.62 GiB | 11.91 B | Vulkan | 999 | tg128 | 9.86 ± 0.01 |
| gemma4 ?B Q4_K - Medium | 6.62 GiB | 11.91 B | Vulkan | 999 | tg256 | 9.82 ± 0.00 |

Commands:

```bash
./build-hip/bin/llama-bench -m mymodels/gemma-4-12b-it-GGUF/gemma-4-12b-it-Q4_K_M.gguf -ngl 999 -p 128 -n 128,256
./build-vulkan/bin/llama-bench -m mymodels/gemma-4-12b-it-GGUF/gemma-4-12b-it-Q4_K_M.gguf -ngl 999 -p 128 -n 128,256
```

`pp128` = prompt processing (128 tokens), `tg128` / `tg256` = text generation. Build: `d403f00ec (9554)`.
