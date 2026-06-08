# mini local agent

A minimal ReAct coding agent: single file, zero third-party dependencies (Python standard library only), connecting to a local `llama-server` (OpenAI-compatible API).

Design goal: faster than Roo / Cline. Local models only do prompt processing at ~40-70 tok/s, and those tools inject tens of thousands of tokens of tool instructions every turn; this agent's system prompt is only a few hundred tokens and it reuses conversation history to hit llama.cpp's prompt cache.

## Prerequisites

1. Python 3.7+
2. A built `llama-server` and a `.gguf` model. The agent **launches `llama-server` for you** on startup (and shuts it down on exit), so you no longer need to start it manually. The defaults point at:
   - server: `C:\Users\bookery\llama.cpp\build-x64-windows-vulkan-release\bin\llama-server.exe`
   - model: `C:\Users\bookery\llama.cpp\mymodels\gemma-4-12b-it-GGUF\gemma-4-12b-it-Q4_K_M.gguf`

   Override either with `--server-bin` / `--model-path`. If a server is already running at the target URL, the agent detects it and reuses it instead of starting a new one. Use `--no-server` to skip launching entirely and connect to an existing server.

## Usage

```powershell
python c:\Users\bookery\mini-agent\agent.py
```

This starts `llama-server`, waits until its `/health` endpoint is ready, then drops you into the task prompt. The server's stdout/stderr is written to `mini-agent\server.log`. Enter a task (empty line to quit), for example:

```
task> Read llama.cpp/src/models/gemma4.cpp and summarize its structure
```

When you quit (empty line / Ctrl+C), the agent terminates the `llama-server` it started.

## Options

| Option | Default | Description |
| --- | --- | --- |
| `--base-url` | built from `--host`/`--port` | OpenAI-compatible endpoint (overrides host/port) |
| `--model` | `local` | Model name (usually ignored by llama-server, any value works) |
| `--yes` | off | Skip confirmation before writing files / running commands |
| `--max-steps` | `20` | Max ReAct steps per task |
| `--max-tokens` | `2048` | Max tokens generated per step (prevents long unresponsive runs) |
| `--think` | off | Enable model reasoning/thinking; disabled by default to avoid stalls |
| `--no-server` | off | Do not launch llama-server; connect to an already-running one |
| `--host` | `127.0.0.1` | llama-server host |
| `--port` | `8080` | llama-server port |
| `--server-bin` | (path above) | Path to the `llama-server` executable |
| `--model-path` | (path above) | Path to the `.gguf` model to serve |
| `--ngl` | `99` | Number of layers to offload to GPU |
| `--ctx-size` | `0` | Context size (`-c`); `0` leaves the server default |
| `--server-arg ARG` | none | Extra raw argument passed to llama-server (repeatable) |
| `--server-timeout` | `300` | Seconds to wait for the server to become ready |

Example (auto-confirm, limit to 10 steps):

```powershell
python c:\Users\bookery\mini-agent\agent.py --yes --max-steps 10
```

Example (serve a different model, pass extra server flags):

```powershell
python c:\Users\bookery\mini-agent\agent.py --model-path C:\models\my.gguf --ctx-size 8192 --server-arg --flash-attn
```

Example (reuse a server you already started elsewhere):

```powershell
python c:\Users\bookery\mini-agent\agent.py --no-server --base-url http://127.0.0.1:8080/v1
```

## Project conventions (memory)

The agent has no long-term memory, so to give it persistent project rules (e.g. "always run the program via `make run`", "compile HIP with `--offload-arch=gfx1150`"), put them in an `AGENTS.md` file in the working directory. On startup the agent auto-loads it and injects it into the system prompt.

- Auto-detected filenames (first match wins): `AGENTS.md`, `AGENT.md`, `.agentrc`, `conventions.md`.
- Override the path with `--notes path\to\file.md`.

This is how you stop the agent from, e.g., invoking `./hip_gemm.exe` directly when you want `make run`.

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

## Local model performance (Vulkan)

Backend model used by the agent: `gemma-4-12b-it-Q4_K_M.gguf` (6.62 GiB, 11.91 B params), measured with `llama-bench`.

Device: AMD Radeon(TM) 890M Graphics (Vulkan, AMD proprietary driver) | uma: 1 | fp16: 1 | bf16: 1 | warp size: 64 | KHR_coopmat

| model | backend | ngl | test | t/s |
| --- | --- | --: | --- | --: |
| gemma4 12B Q4_K - Medium | Vulkan | 99 | pp128 | 82.98 ± 0.97 |
| gemma4 12B Q4_K - Medium | Vulkan | 99 | tg64 | 8.85 ± 0.02 |

Command:

```powershell
build-x64-windows-vulkan-release\bin\llama-bench.exe -m .\mymodels\gemma-4-12b-it-GGUF\gemma-4-12b-it-Q4_K_M.gguf -ngl 99 -p 128 -n 64
```

`pp128` = prompt processing (128 tokens), `tg64` = text generation (64 tokens). Build: `0dbfa66a1 (9512)`.

## Local model performance (ROCm/HIP)

Backend model used by the agent: `gemma-4-12b-it-Q4_K_M.gguf` (6.62 GiB, 11.91 B params), measured with `llama-bench`.

| model | size | params | backend | ngl | test | t/s |
| --- | --: | --: | --- | --: | --- | --: |
| gemma4 12B Q4_K - Medium | 6.62 GiB | 11.91 B | ROCm | 999 | pp128 | 58.63 +/- 0.10 |
| gemma4 12B Q4_K - Medium | 6.62 GiB | 11.91 B | ROCm | 999 | pp512 | 57.64 +/- 0.49 |
| gemma4 12B Q4_K - Medium | 6.62 GiB | 11.91 B | ROCm | 999 | pp2048 | 52.87 +/- 0.25 |
| gemma4 12B Q4_K - Medium | 6.62 GiB | 11.91 B | ROCm | 999 | tg128 | 9.19 +/- 0.01 |
| gemma4 12B Q4_K - Medium | 6.62 GiB | 11.91 B | ROCm | 999 | tg256 | 9.12 +/- 0.01 |
