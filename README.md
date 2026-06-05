# mini local agent

A minimal ReAct coding agent: single file, zero third-party dependencies (Python standard library only), connecting to a local `llama-server` (OpenAI-compatible API).

Design goal: faster than Roo / Cline. Local models only do prompt processing at ~40-70 tok/s, and those tools inject tens of thousands of tokens of tool instructions every turn; this agent's system prompt is only a few hundred tokens and it reuses conversation history to hit llama.cpp's prompt cache.

## Prerequisites

1. Python 3.7+
2. A running llama-server, for example:

```powershell
.\build-x64-windows-vulkan-release\bin\llama-server.exe -m .\mymodels\gemma-4-12b-it-GGUF\gemma-4-12b-it-Q4_K_M.gguf -ngl 99 --host 127.0.0.1 --port 8080
```

## Usage

```powershell
python c:\Users\bookery\mini-agent\agent.py
```

After it starts, enter a task (empty line to quit), for example:

```
task> Read llama.cpp/src/models/gemma4.cpp and summarize its structure
```

## Options

| Option | Default | Description |
| --- | --- | --- |
| `--base-url` | `http://127.0.0.1:8080/v1` | OpenAI-compatible endpoint |
| `--model` | `local` | Model name (usually ignored by llama-server, any value works) |
| `--yes` | off | Skip confirmation before writing files / running commands |
| `--max-steps` | `20` | Max ReAct steps per task |
| `--max-tokens` | `2048` | Max tokens generated per step (prevents long unresponsive runs) |
| `--think` | off | Enable model reasoning/thinking; disabled by default to avoid stalls |

Example (auto-confirm, limit to 10 steps):

```powershell
python c:\Users\bookery\mini-agent\agent.py --yes --max-steps 10
```

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
