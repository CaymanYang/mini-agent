#!/usr/bin/env python3
"""mini local agent: a tiny ReAct coding agent for an OpenAI-compatible local server (llama.cpp)."""

import argparse
import atexit
import difflib
import glob
import json
import os
import queue
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error

# Windows consoles often default to cp1252 which cannot encode CJK; force UTF-8.
for _stream in (sys.stdout, sys.stderr, sys.stdin):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

MAX_OBS_CHARS = 6000  # cap observation size fed back into context to keep prompts small
MAX_READ_CHARS = 8000

# Defaults for auto-launching llama-server. Override via CLI flags.
DEFAULT_MODEL_REL = os.path.join(
    "models", "gemma-4-12b-it-GGUF", "gemma-4-12b-it-Q4_K_M.gguf"
)
MODEL_DIR_NAMES = ("models", "mymodel", "mymodels")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_NGL = 99
BACKEND_BUILD_DIRS = {
    "vulkan": ["build-vulkan", "build-x64-windows-vulkan-release"],
    "hip": ["build-hip"],
}
DEFAULT_CONTEXT_LIMIT = 8192
DEFAULT_COMPACT_THRESHOLD = 0.70
DEFAULT_COMPACT_KEEP_MESSAGES = 6
DEFAULT_COMPACT_MAX_TOKENS = 1024
DEFAULT_MEMORY_FILE = os.path.join(".mini-agent", "memory", "session.md")
THINKING_HEARTBEAT_SECONDS = 5

SYSTEM_PROMPT = """You are a minimal coding agent. Solve the user's task one tool call at a time.

Output exactly:
ACTION
{"tool": "<name>", "args": { ... }}

Tools:
- list_dir: {"path": str optional}
- read_file: {"path": str}
- str_replace: {"path": str, "old_string": str, "new_string": str}
- write_file: {"path": str, "content": str, "overwrite": bool optional}
- open_file: {"path": str, "line": int optional}
- run_shell: {"command": str}
- mcp_call: {"server": str, "tool": str, "args": object optional}
- finish: {"answer": str}

Rules:
- Emit one ACTION JSON only; never use native tool_call or OpenAI tool_calls syntax.
- Put all tool parameters inside args, including "overwrite": true.
- JSON must be valid; escape newlines in strings.
- Use relative paths; list_dir/read_file before guessing.
- Edit existing files with str_replace only: copy an exact unique old_string from read_file output, without line numbers.
- Use write_file only for new files.
- If the user asks to open a file, call open_file.
- If the user asks about git status/diff/log, call run_shell with the git command.
- If the user asks to commit or push, call run_shell with git after confirming intent.
- For git commits, write a concise message about behavior changed, not file names.
- Do not claim files were changed unless the latest OBSERVATION says OK.
- After each OBSERVATION, decide the next ACTION. When done, call finish."""


CONVENTION_FILENAMES = ["AGENTS.md", "AGENT.md", ".agentrc", "conventions.md"]


def load_conventions(explicit_path=None):
    """Load project-specific conventions to inject into the system prompt.

    Looks for an explicit --notes file first, then well-known filenames in the
    working directory. This gives the agent persistent project rules (e.g. "use
    make run", "compile with --offload-arch=gfx1150") it would otherwise forget.
    """
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    candidates += [os.path.join(os.getcwd(), name) for name in CONVENTION_FILENAMES]
    for path in candidates:
        if path and os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read().strip()
            except OSError:
                continue
            if text:
                return path, text
    return None, None


def _unique_existing_dirs(paths):
    result = []
    seen = set()
    for path in paths:
        if not path:
            continue
        path = os.path.abspath(os.path.expanduser(path))
        key = os.path.normcase(path)
        if key not in seen and os.path.isdir(path):
            seen.add(key)
            result.append(path)
    return result


def find_llama_roots():
    """Return likely llama.cpp checkout locations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    home = os.path.expanduser("~")
    candidates = [
        os.environ.get("LLAMA_CPP_DIR"),
        script_dir,
        os.getcwd(),
        os.path.join(os.getcwd(), "llama.cpp"),
        os.path.join(os.path.dirname(script_dir), "llama.cpp"),
        os.path.join(home, "llama.cpp"),
    ]
    return _unique_existing_dirs(candidates)


def _server_executable_name():
    return "llama-server.exe" if os.name == "nt" else "llama-server"


def _backend_order(backend):
    if backend != "auto":
        return [backend]
    # Vulkan is the portable default. HIP can still be selected explicitly.
    return ["vulkan", "hip"]


def _unique_files(paths):
    result = []
    seen = set()
    for path in paths:
        path = os.path.abspath(os.path.expanduser(path))
        key = os.path.normcase(path)
        if key not in seen and os.path.isfile(path):
            seen.add(key)
            result.append(path)
    return result


def _is_llm_model(path):
    name = os.path.basename(path).lower()
    return not (name.startswith("ggml-vocab-") or name.startswith("mmproj-"))


def resolve_server_bin(explicit_path, backend):
    """Find a llama-server binary for the selected backend."""
    if explicit_path:
        return os.path.abspath(os.path.expanduser(explicit_path))
    if backend == "hip" and os.name == "nt":
        raise RuntimeError("ROCm/HIP backend is Linux-only; use --backend vulkan on Windows")

    exe = _server_executable_name()
    searched = []
    for root in find_llama_roots():
        for name in _backend_order(backend):
            if name == "hip" and os.name == "nt":
                continue
            for build_dir in BACKEND_BUILD_DIRS[name]:
                path = os.path.join(root, build_dir, "bin", exe)
                searched.append(path)
                if os.path.isfile(path):
                    return path
    raise RuntimeError(
        "llama-server not found for backend "
        f"{backend!r}; pass --server-bin or build llama.cpp first. Searched:\n"
        + "\n".join(f"  - {path}" for path in searched)
    )


def find_model_paths():
    """Find the default GGUF model near a llama.cpp checkout."""
    matches = []
    searched = []
    for root in find_llama_roots():
        preferred = os.path.join(root, DEFAULT_MODEL_REL)
        searched.append(preferred)
        if os.path.isfile(preferred):
            matches.append(preferred)

        for dirname in MODEL_DIR_NAMES:
            pattern = os.path.join(root, dirname, "**", "*.[gG][gG][uU][fF]")
            searched.append(pattern)
            matches.extend(path for path in sorted(glob.glob(pattern, recursive=True))
                           if _is_llm_model(path))

    return _unique_files(matches), searched


def choose_model_path(candidates):
    if len(candidates) == 1 or not sys.stdin.isatty():
        return candidates[0]

    print("\nAvailable models:")
    cwd = os.getcwd()
    for i, path in enumerate(candidates, 1):
        shown = os.path.relpath(path, cwd) if path.startswith(cwd + os.sep) else path
        print(f"  {i}. {shown}")
    while True:
        choice = input(f"Choose model [1-{len(candidates)}] (default 1): ").strip()
        if not choice:
            return candidates[0]
        if choice.isdigit() and 1 <= int(choice) <= len(candidates):
            return candidates[int(choice) - 1]
        print("Invalid choice.")


def resolve_model_path(explicit_path, prompt=True):
    if explicit_path:
        return os.path.abspath(os.path.expanduser(explicit_path))

    candidates, searched = find_model_paths()
    if candidates:
        return choose_model_path(candidates) if prompt else candidates[0]

    raise RuntimeError(
        "model file not found; pass --model-path. Searched:\n"
        + "\n".join(f"  - {path}" for path in searched)
    )


def _server_root(base_url):
    """Strip a trailing /v1 so we can hit llama.cpp's /health endpoint."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return root


def server_is_up(base_url, timeout=2.0):
    """Return True if a llama.cpp server already answers at base_url."""
    health = _server_root(base_url) + "/health"
    try:
        with urllib.request.urlopen(health, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


class TokenStats:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.prompt = 0
        self.completion = 0
        self.total = 0
        self.steps = 0

    def add(self, usage):
        if not self.enabled or not usage:
            return
        prompt = usage.get("prompt_tokens") or usage.get("prompt") or 0
        completion = usage.get("completion_tokens") or usage.get("completion") or 0
        total = usage.get("total_tokens") or usage.get("total") or prompt + completion
        self.prompt += int(prompt)
        self.completion += int(completion)
        self.total += int(total)
        self.steps += 1

    def summary(self):
        return f"prompt={self.prompt}, completion={self.completion}, total={self.total}"


def tokenize_text(base_url, text, timeout=10.0):
    """Return llama-server token count for text, or None if unavailable."""
    url = _server_root(base_url) + "/tokenize"
    payload = json.dumps({"content": text}).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    tokens = body.get("tokens")
    return len(tokens) if isinstance(tokens, list) else None


def estimate_message_tokens(base_url, messages):
    """Approximate context tokens by tokenizing message text without chat template."""
    chunks = []
    for msg in messages:
        content = msg.get("content", "")
        if content:
            chunks.append(f"{msg.get('role', 'user')}:\n{content}")
    if not chunks:
        return 0
    return tokenize_text(base_url, "\n\n".join(chunks))


def load_memory(memory_file):
    if not memory_file:
        return None
    path = os.path.abspath(os.path.expanduser(memory_file))
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read().strip()
    except OSError:
        return None
    return text or None


def save_memory(memory_file, text):
    path = os.path.abspath(os.path.expanduser(memory_file))
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    memory_root = os.path.dirname(parent)
    if os.path.basename(memory_root) == ".mini-agent":
        ignore_path = os.path.join(memory_root, ".gitignore")
        if not os.path.exists(ignore_path):
            with open(ignore_path, "w", encoding="utf-8") as f:
                f.write("*\n!.gitignore\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")
    return path


def messages_to_transcript(messages):
    chunks = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if content:
            chunks.append(f"## {role}\n{content}")
    return "\n\n".join(chunks)


COMPACT_PROMPT = """Summarize this mini-agent session into concise long-term memory.

Rules:
- Only use facts from the transcript.
- Do not invent files, commands, or decisions.
- Preserve details needed to continue the task.
- Keep it short but actionable.

Write markdown with exactly these sections:
# Session Summary

## Goal

## Current State

## Important Decisions

## Files Touched

## Commands Run

## Open Tasks

## Constraints
"""


def compact_messages(base_url, model, messages, memory_file, keep_messages,
                     max_tokens, think, stall_timeout):
    """Summarize older history, save it, and replace it with a compact memory."""
    if len(messages) <= keep_messages + 1:
        return False
    system_msg = messages[0]
    recent = messages[-keep_messages:] if keep_messages > 0 else []
    transcript = messages_to_transcript(messages[1:])
    compact_request = [
        {"role": "system", "content": COMPACT_PROMPT},
        {"role": "user", "content": transcript},
    ]
    print(f"[auto-compact] summarizing history into {memory_file}")
    summary, _, _ = call_llm(base_url, model, compact_request,
                             max_tokens=max_tokens, think=think,
                             stop_on_action=False,
                             stall_timeout=stall_timeout,
                             show_tokens=False)
    summary = summary.strip()
    if not summary:
        print("[auto-compact] skipped: model returned empty summary")
        return False
    path = save_memory(memory_file, summary)
    memory_msg = {
        "role": "user",
        "content": f"Compressed session memory loaded from {path}:\n\n{summary}",
    }
    messages[:] = [system_msg, memory_msg] + recent
    print(f"[auto-compact] wrote {path}; kept last {len(recent)} messages")
    return True


def start_server(server_bin, model_path, host, port, ngl, ctx_size, extra_args, log_path):
    """Launch llama-server as a child process; return the Popen handle."""
    if not os.path.isfile(server_bin):
        raise RuntimeError(f"llama-server not found: {server_bin} (pass --server-bin)")
    if not os.path.isfile(model_path):
        raise RuntimeError(f"model file not found: {model_path} (pass --model-path)")
    cmd = [
        server_bin,
        "-m", model_path,
        "--host", host,
        "--port", str(port),
        "-ngl", str(ngl),
    ]
    if ctx_size:
        cmd += ["-c", str(ctx_size)]
    if extra_args:
        cmd += extra_args
    print(f"[starting llama-server] {' '.join(cmd)}")
    print(f"[server log] {log_path}")
    log = open(log_path, "w", encoding="utf-8", errors="replace")
    # New process group so we can terminate it cleanly on Windows and POSIX.
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, **kwargs)
    proc._log_handle = log  # keep a reference so it isn't garbage collected
    return proc


def wait_for_server(base_url, proc, timeout=300):
    """Poll /health until the server is ready, or fail if it exits early."""
    deadline = time.time() + timeout
    print("[waiting for server to be ready] ", end="", flush=True)
    while time.time() < deadline:
        if proc.poll() is not None:
            print()
            raise RuntimeError(
                f"llama-server exited early (code {proc.returncode}); "
                f"check the server log for details."
            )
        if server_is_up(base_url, timeout=2.0):
            print(" ready")
            return True
        print(".", end="", flush=True)
        time.sleep(1.0)
    print()
    raise RuntimeError(f"server did not become ready within {timeout}s")


def stop_server(proc):
    """Terminate the llama-server child process and close its log file."""
    if proc is None or proc.poll() is not None:
        return
    print("\n[stopping llama-server]")
    try:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    except Exception:
        pass
    finally:
        handle = getattr(proc, "_log_handle", None)
        if handle:
            try:
                handle.close()
            except Exception:
                pass


def _reasoning_text(delta_or_message):
    texts = []
    for key in ("reasoning_content", "reasoning", "thinking", "thought", "reasoning_text"):
        value = delta_or_message.get(key)
        if isinstance(value, str):
            texts.append(value)
        elif isinstance(value, dict):
            for subkey in ("content", "text", "summary"):
                subvalue = value.get(subkey)
                if isinstance(subvalue, str):
                    texts.append(subvalue)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict):
                    for subkey in ("content", "text", "summary"):
                        subvalue = item.get(subkey)
                        if isinstance(subvalue, str):
                            texts.append(subvalue)
    return "".join(texts)


def call_llm(base_url, model, messages, max_tokens=2048, think=False, stream=True,
             stop_on_action=True, stall_timeout=180, show_tokens=True):
    """Call /chat/completions. Streams tokens to stdout and returns the full assistant text."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "temperature": 0.3,
        # Bound each step: without this the server uses n_predict=-1 (unlimited),
        # which lets a reasoning model run for thousands of tokens and appear stuck.
        "max_tokens": max_tokens,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    if not think:
        # This Gemma "thinking" model otherwise spends thousands of tokens in a
        # hidden reasoning_content stream (no visible output -> looks frozen).
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )

    try:
        resp = urllib.request.urlopen(req, timeout=max(stall_timeout * 2, 60))
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to connect to local model server {url}: {e}")

    if not stream:
        body = json.loads(resp.read().decode("utf-8"))
        choice = body["choices"][0]
        message = choice["message"]
        action = _action_from_tool_calls(message.get("tool_calls") or [])
        text = _format_action_text(action) if action else (message.get("content") or "")
        reasoning = _reasoning_text(message)
        if think and reasoning:
            print(f"[thinking] {reasoning}")
        print(text, end="", flush=True)
        print()
        return text, choice.get("finish_reason"), body.get("usage")

    # Read the HTTP stream in a background thread and hand lines to the main thread
    # via a queue. On Windows a blocking socket read does NOT let Python deliver
    # KeyboardInterrupt until the read returns; polling the queue with a short
    # timeout keeps the main thread responsive so Ctrl+C works immediately.
    q = queue.Queue(maxsize=10000)
    stop = threading.Event()

    def _reader():
        try:
            for raw in resp:
                if stop.is_set():
                    break
                q.put(raw)
        except Exception as e:  # noqa: BLE001 - socket closed/aborted surfaces here
            q.put(e)
        finally:
            q.put(None)  # sentinel: stream finished

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()

    parts = []
    in_reasoning = False
    finish_reason = None
    usage = None
    streamed_tokens = 0
    last_data = time.time()
    last_visible = time.time()
    tool_calls = {}
    printed_tool_call = False
    printed_waiting = False
    try:
        while True:
            try:
                raw = q.get(timeout=0.2)
            except queue.Empty:
                # Inactivity watchdog: abort if the model produces nothing for a
                # while (stalled/busy server) instead of hanging on a fixed total
                # timeout. Healthy streaming resets last_data on every chunk.
                if time.time() - last_data > stall_timeout:
                    raise RuntimeError(
                        f"no output from the model for {stall_timeout}s "
                        "(server may be stalled or busy; try restarting it)")
                if think and time.time() - last_visible > THINKING_HEARTBEAT_SECONDS:
                    if in_reasoning:
                        print(" ...", end="", flush=True)
                    else:
                        print("[thinking...] ", end="", flush=True)
                        printed_waiting = True
                    last_visible = time.time()
                continue  # also gives the main thread a chance to see Ctrl+C
            last_data = time.time()
            if raw is None:
                break
            if isinstance(raw, Exception):
                raise RuntimeError(f"stream read failed: {raw}")
            line = raw.decode("utf-8").strip()
            if not line or not line.startswith("data:"):
                continue
            chunk = line[len("data:"):].strip()
            if chunk == "[DONE]":
                break
            try:
                obj = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
            delta = choice.get("delta", {})
            delta_tool_calls = delta.get("tool_calls") or []
            if delta_tool_calls:
                if in_reasoning:
                    print("\n", end="", flush=True)
                    in_reasoning = False
                if not printed_tool_call:
                    if printed_waiting:
                        print("\n", end="", flush=True)
                        printed_waiting = False
                    print("[tool_call]", end="", flush=True)
                    printed_tool_call = True
                    last_visible = time.time()
                streamed_tokens += 1
                for delta_call in delta_tool_calls:
                    idx = delta_call.get("index", 0)
                    call = tool_calls.setdefault(idx, {"function": {"name": "", "arguments": ""}})
                    if delta_call.get("id"):
                        call["id"] = delta_call["id"]
                    if delta_call.get("type"):
                        call["type"] = delta_call["type"]
                    function = delta_call.get("function") or {}
                    target = call.setdefault("function", {"name": "", "arguments": ""})
                    if function.get("name"):
                        target["name"] += function["name"]
                    if function.get("arguments"):
                        target["arguments"] += function["arguments"]
            # Reasoning tokens arrive separately; show them so output never looks frozen.
            reasoning = _reasoning_text(delta)
            if reasoning:
                if printed_waiting:
                    print("\n", end="", flush=True)
                    printed_waiting = False
                if not in_reasoning:
                    print("[thinking] ", end="", flush=True)
                    in_reasoning = True
                print(reasoning, end="", flush=True)
                streamed_tokens += 1
                last_visible = time.time()
            piece = delta.get("content")
            if piece:
                if printed_waiting:
                    print("\n", end="", flush=True)
                    printed_waiting = False
                if in_reasoning:
                    print("\n", end="", flush=True)
                    in_reasoning = False
                parts.append(piece)
                print(piece, end="", flush=True)
                streamed_tokens += 1
                last_visible = time.time()
            if piece:
                # Early stop: once a full ACTION is available, don't wait for the
                # model to keep rambling (it often appends text up to max_tokens).
                if stop_on_action and "}" in piece:
                    joined = "".join(parts)
                    if "ACTION" in joined and parse_action(joined) is not None:
                        finish_reason = "stop"  # complete action; not truncated
                        break
    finally:
        # Unblock and tear down the reader thread (closing the socket aborts its read).
        stop.set()
        try:
            resp.close()
        except Exception:
            pass
    print()
    if usage is None and streamed_tokens:
        usage = {"completion_tokens": streamed_tokens, "total_tokens": streamed_tokens, "estimated": True}
    if not parts and tool_calls:
        action = _action_from_tool_calls([tool_calls[i] for i in sorted(tool_calls)])
        if action:
            return _format_action_text(action), finish_reason, usage
    return "".join(parts), finish_reason, usage


def list_dir(args):
    path = args.get("path") or "."
    if not os.path.isdir(path):
        return f"ERROR: not a directory: {path}"
    entries = []
    for name in sorted(os.listdir(path)):
        is_dir = os.path.isdir(os.path.join(path, name))
        entries.append(name + ("/" if is_dir else ""))
    listing = "\n".join(entries) if entries else "(empty)"
    return f"{os.path.abspath(path)}\n{listing}"


def read_file(args):
    path = args["path"]
    if not os.path.exists(path):
        return f"ERROR: file not found: {path}"
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    truncated = ""
    if len(content) > MAX_READ_CHARS:
        content = content[:MAX_READ_CHARS]
        truncated = f"\n... [truncated, showing first {MAX_READ_CHARS} chars]"
    numbered = "\n".join(
        f"{i+1:6}|{line}" for i, line in enumerate(content.splitlines())
    )
    return numbered + truncated


def open_file(args):
    path = args["path"]
    line = args.get("line")
    if not os.path.exists(path):
        return f"ERROR: file not found: {path}"

    abs_path = os.path.abspath(path)
    try:
        line = int(line) if line is not None else None
    except (TypeError, ValueError):
        line = None

    code_bin = shutil.which("code")
    if code_bin:
        target = f"{abs_path}:{line}" if line and line > 0 else abs_path
        try:
            subprocess.Popen(
                [code_bin, "-g", target],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return f"OK: opened {target}"
        except OSError as e:
            return f"ERROR: failed to open editor: {e}"

    if line and line > 0:
        return f"Open manually: vim +{line} {shlex.quote(abs_path)}"
    return f"Open manually: vim {shlex.quote(abs_path)}"


def write_file(args, auto_yes):
    path = args["path"]
    content = args.get("content", "")
    if os.path.exists(path) and not args.get("overwrite"):
        # Steer the model toward editing instead of rewriting existing files.
        return (f"ERROR: {path} already exists. To EDIT an existing file use "
                "str_replace. To replace the entire file anyway, pass "
                'valid JSON args with "overwrite": true.')
    print(f"\n--- about to write file: {path} ({len(content)} chars) ---")
    if not auto_yes and not _confirm():
        return "SKIPPED: user declined write_file"
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    abs_path = os.path.abspath(path)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        written = f.read()
    verified = written == content
    size = os.path.getsize(path)
    if not verified:
        return f"ERROR: write_file verification failed for {abs_path}"
    return (f"OK: write_file succeeded; verified=true; path={abs_path}; "
            f"chars={len(content)}; bytes={size}")


def _strip_read_file_line_numbers(text):
    return "\n".join(re.sub(r"^\s*\d+\|", "", line) for line in text.splitlines())


def _str_replace_old_variants(old):
    variants = []

    def add(value):
        if value not in variants:
            variants.append(value)

    add(old)
    stripped = _strip_read_file_line_numbers(old)
    if stripped != old:
        add(stripped)
    for value in list(variants):
        add(value.replace("\r\n", "\n"))
        add(value.replace("\n", "\r\n"))
    return variants


def _nearby_old_string_hints(content, old, max_hints=3):
    lines = content.splitlines()
    if not lines:
        return ""

    old_lines = [line.strip() for line in _strip_read_file_line_numbers(old).splitlines()
                 if line.strip()]
    query = max(old_lines, key=len, default=old.strip())
    if not query:
        return ""

    scored = []
    for i, line in enumerate(lines):
        score = difflib.SequenceMatcher(None, query, line.strip()).ratio()
        if query in line:
            score = max(score, 0.95)
        scored.append((score, i))
    scored.sort(reverse=True)

    snippets = []
    used = set()
    for score, i in scored:
        if score < 0.35:
            break
        start = max(0, i - 2)
        end = min(len(lines), i + 3)
        key = (start, end)
        if key in used:
            continue
        used.add(key)
        snippet = "\n".join(lines[start:end])
        if snippet:
            snippets.append(snippet)
        if len(snippets) >= max_hints:
            break

    if not snippets:
        return ""
    text = "\n\n".join(f"--- candidate {i + 1} ---\n{snippet}"
                       for i, snippet in enumerate(snippets))
    return "\n\nNearby exact snippets from the current file:\n" + text[:2000]


def str_replace(args, auto_yes):
    path = args["path"]
    old = args["old_string"]
    new = args.get("new_string", "")
    if not os.path.exists(path):
        return f"ERROR: file not found: {path}"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    matched_old = None
    count = 0
    for candidate in _str_replace_old_variants(old):
        count = content.count(candidate)
        if count:
            matched_old = candidate
            break
    if count == 0:
        return (
            "ERROR: old_string not found.\n"
            "Read the file again or copy a shorter exact snippet from below into old_string.\n"
            "Do not rewrite from memory."
            + _nearby_old_string_hints(content, old)
        )
    if count > 1:
        return (f"ERROR: old_string matched {count} times; it must be unique. "
                "Include more surrounding context.")
    print(f"\n--- about to edit file: {path} (1 replacement, {len(matched_old)}->{len(new)} chars) ---")
    if not auto_yes and not _confirm():
        return "SKIPPED: user declined str_replace"
    updated = content.replace(matched_old, new)
    with open(path, "w", encoding="utf-8") as f:
        f.write(updated)
    abs_path = os.path.abspath(path)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        written = f.read()
    if written != updated:
        return f"ERROR: str_replace verification failed for {abs_path}"
    return (f"OK: str_replace succeeded; verified=true; path={abs_path}; "
            f"replacements=1; chars={len(written)}")


def _windows_shell():
    # Prefer PowerShell 7 (pwsh) which supports && / ||; fall back to Windows
    # PowerShell 5.1 (powershell) which does not.
    return "pwsh" if shutil.which("pwsh") else "powershell"


def run_shell(args, auto_yes):
    command = args["command"]
    print(f"\n--- about to run command: {command} ---")
    if re.search(r"\bgit\s+commit\b", command):
        print("[commit message] Prefer a concise behavior summary, not just file names.")
    if not auto_yes and not _confirm():
        return "SKIPPED: user declined run_shell"
    is_windows = os.name == "nt"
    if is_windows:
        full = [_windows_shell(), "-NoProfile", "-Command", command]
    else:
        full = ["bash", "-lc", command]
    try:
        proc = subprocess.run(
            full, capture_output=True, text=True, timeout=120
        )
    except subprocess.TimeoutExpired:
        return "ERROR: command timed out after 120s"
    out = (proc.stdout or "") + (proc.stderr or "")
    if len(out) > MAX_OBS_CHARS:
        out = out[:MAX_OBS_CHARS] + "\n... [output truncated]"
    return f"exit_code={proc.returncode}\n{out}"


def _gh_auth_token():
    if shutil.which("gh") is None:
        return None
    try:
        proc = subprocess.run(["gh", "auth", "token"], capture_output=True,
                              text=True, timeout=10)
    except Exception:
        return None
    token = proc.stdout.strip()
    return token if proc.returncode == 0 and token else None


class McpServer:
    def __init__(self, name, spec):
        self.name = name
        self.spec = spec
        self.proc = None
        self.next_id = 1
        self.responses = queue.Queue()
        self.tools = []

    def start(self):
        command = self.spec.get("command")
        if not command:
            raise RuntimeError(f"MCP server {self.name}: missing command")
        args = self.spec.get("args", [])
        if isinstance(command, str) and not args:
            cmd = shlex.split(command)
        else:
            cmd = [command] + list(args)

        env = os.environ.copy()
        env.update({str(k): str(v) for k, v in self.spec.get("env", {}).items()})
        joined = " ".join(cmd)
        if ("server-github" in joined or self.name.lower() == "github"):
            if not env.get("GITHUB_PERSONAL_ACCESS_TOKEN") and not env.get("GITHUB_TOKEN"):
                token = _gh_auth_token()
                if token:
                    env["GITHUB_PERSONAL_ACCESS_TOKEN"] = token

        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            bufsize=1,
        )
        threading.Thread(target=self._reader, daemon=True).start()
        self._request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "mini-agent", "version": "0.1"},
        })
        self._notify("notifications/initialized", {})
        listed = self._request("tools/list", {})
        self.tools = listed.get("tools", [])

    def _reader(self):
        while self.proc and self.proc.stdout:
            line = self.proc.stdout.readline()
            if not line:
                break
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in msg:
                self.responses.put(msg)

    def _send(self, msg):
        if not self.proc or self.proc.poll() is not None:
            raise RuntimeError(f"MCP server {self.name} is not running")
        self.proc.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()

    def _notify(self, method, params):
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _request(self, method, params, timeout=30):
        req_id = self.next_id
        self.next_id += 1
        self._send({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        deadline = time.time() + timeout
        skipped = []
        while time.time() < deadline:
            try:
                msg = self.responses.get(timeout=0.2)
            except queue.Empty:
                continue
            if msg.get("id") != req_id:
                skipped.append(msg)
                continue
            for item in skipped:
                self.responses.put(item)
            if "error" in msg:
                raise RuntimeError(f"MCP {self.name} {method} failed: {msg['error']}")
            return msg.get("result", {})
        for item in skipped:
            self.responses.put(item)
        raise RuntimeError(f"MCP {self.name} {method} timed out")

    def call_tool(self, tool, args):
        return self._request("tools/call", {"name": tool, "arguments": args or {}}, timeout=120)

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass


class McpManager:
    def __init__(self):
        self.servers = {}

    def load_config(self, path):
        if not path:
            return
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isfile(path):
            raise RuntimeError(f"MCP config not found: {path}")
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            config = json.load(f)
        servers = config.get("mcpServers", config.get("servers", {}))
        for name, spec in servers.items():
            server = McpServer(name, spec)
            server.start()
            self.servers[name] = server

    def describe_tools(self):
        lines = []
        for name, server in self.servers.items():
            for tool in server.tools:
                desc = (tool.get("description") or "").strip().replace("\n", " ")
                lines.append(f"- {name}.{tool.get('name')}: {desc}")
        return "\n".join(lines)

    def call(self, args):
        server_name = args["server"]
        tool_name = args["tool"]
        if server_name not in self.servers:
            return f"ERROR: unknown MCP server '{server_name}'"
        result = self.servers[server_name].call_tool(tool_name, args.get("args", {}))
        text = json.dumps(result, ensure_ascii=False, indent=2)
        if len(text) > MAX_OBS_CHARS:
            text = text[:MAX_OBS_CHARS] + "\n... [truncated]"
        return text

    def stop(self):
        for server in self.servers.values():
            server.stop()


def _confirm():
    try:
        ans = input("Confirm? [y/N] ").strip().lower()
    except EOFError:
        return False
    return ans in ("y", "yes")


def _json_loads_relaxed(candidate):
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(candidate, strict=False)
    except json.JSONDecodeError:
        pass

    # Models frequently emit raw Windows paths like ".\WorkDir\app.exe".
    # Backslashes such as \W or \h are invalid JSON escapes, so escape any
    # backslash that isn't part of a valid escape and try once more.
    escaped = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", candidate)
    try:
        return json.loads(escaped)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(escaped, strict=False)
    except json.JSONDecodeError:
        pass

    # Gemma-style text sometimes uses JavaScript-like object keys.
    quoted_keys = re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_-]*)(\s*:)', r'\1"\2"\3', escaped)
    try:
        return json.loads(quoted_keys)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(quoted_keys, strict=False)
    except json.JSONDecodeError:
        return None


def _normalize_action_object(obj):
    if not isinstance(obj, dict):
        return None

    if "tool" in obj:
        args = obj.get("args")
        if not isinstance(args, dict):
            # Some models flatten args to the top level instead of nesting them
            # under "args" (e.g. {"tool":"run_shell","command":"ls"}). Accept that.
            args = {k: v for k, v in obj.items() if k != "tool"}
        return {"tool": obj["tool"], "args": args}

    name = obj.get("name") or obj.get("function")
    if isinstance(name, dict):
        name = name.get("name")
    if not isinstance(name, str):
        return None

    args = obj.get("args", obj.get("arguments", {}))
    if isinstance(args, str):
        args = _json_loads_relaxed(args) or {}
    if not isinstance(args, dict):
        args = {}

    return _action_from_tool_name(name, args)


def _action_from_tool_name(name, args):
    if "." in name and name not in {"list_dir", "read_file", "str_replace", "write_file", "run_shell", "mcp_call", "finish"}:
        server, tool = name.split(".", 1)
        return {"tool": "mcp_call", "args": {"server": server, "tool": tool, "args": args}}
    return {"tool": name, "args": args}


def _action_from_tool_calls(tool_calls):
    if not tool_calls:
        return None
    call = tool_calls[0]
    function = call.get("function") or {}
    name = function.get("name") or call.get("name")
    if not name:
        return None
    args = function.get("arguments", call.get("arguments", {}))
    if isinstance(args, str):
        args = _json_loads_relaxed(args) or {}
    if not isinstance(args, dict):
        args = {}
    return _action_from_tool_name(name, args)


def _format_action_text(action):
    return "ACTION\n" + json.dumps(action, ensure_ascii=False)


def _loose_string_value(text, key):
    match = re.search(rf'"{re.escape(key)}"\s*:\s*"', text)
    if not match:
        return None

    out = []
    i = match.end()
    while i < len(text):
        ch = text[i]
        if ch == '"':
            rest = text[i + 1:].lstrip()
            if not rest or rest[0] in ",}":
                return "".join(out)
            out.append(ch)
            i += 1
            continue
        if ch == "\\" and i + 1 < len(text):
            nxt = text[i + 1]
            if nxt in "\r\n":
                if nxt == "\r" and i + 2 < len(text) and text[i + 2] == "\n":
                    i += 3
                else:
                    i += 2
                out.append("\n")
                continue
            escapes = {
                '"': '"',
                "\\": "\\",
                "/": "/",
                "b": "\b",
                "f": "\f",
                "n": "\n",
                "r": "\r",
                "t": "\t",
            }
            if nxt in escapes:
                out.append(escapes[nxt])
                i += 2
                continue
        out.append(ch)
        i += 1
    return None


def _loose_bool_value(text, key):
    match = re.search(rf'"?{re.escape(key)}"?\s*:\s*(true|false)', text, re.IGNORECASE)
    if not match:
        return None
    return match.group(1).lower() == "true"


def _parse_loose_action_object(text):
    tool = _loose_string_value(text, "tool")
    if not tool:
        return None

    args = {}
    for key in ("path", "old_string", "new_string", "content", "command",
                "server", "answer"):
        value = _loose_string_value(text, key)
        if value is not None:
            args[key] = value
    overwrite = _loose_bool_value(text, "overwrite")
    if overwrite is not None:
        args["overwrite"] = overwrite

    if not args and tool not in {"list_dir"}:
        return None
    return {"tool": tool, "args": args}


def _parse_gemma_tool_call(text):
    match = re.search(r"<\|tool_call\>(.*?)<tool_call\|>", text, re.DOTALL)
    if not match:
        return None

    body = match.group(1).strip()
    if body.startswith("call:"):
        body = body[len("call:"):].strip()
    if body.startswith("tool:"):
        body = body[len("tool:"):].strip()

    brace = body.find("{")
    if brace == -1:
        return None

    name = body[:brace].strip(" \t\r\n:") or None
    candidate = _extract_balanced(body[brace:])
    if candidate is None:
        return None

    obj = _json_loads_relaxed(candidate)
    if obj is None:
        obj = _parse_loose_action_object(candidate)
    if obj is None:
        return None

    action = _normalize_action_object(obj)
    if action:
        return action
    if name:
        return _action_from_tool_name(name, obj if isinstance(obj, dict) else {})
    return None


def parse_action(text):
    """Extract the JSON action object following an ACTION marker (or the first JSON object)."""
    gemma_action = _parse_gemma_tool_call(text)
    if gemma_action:
        return gemma_action

    idx = text.find("ACTION")
    search_from = idx + len("ACTION") if idx != -1 else 0
    snippet = text[search_from:]

    # Prefer a fenced block if present.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", snippet, re.DOTALL)
    candidate = fence.group(1) if fence else None

    if candidate is None:
        brace = snippet.find("{")
        if brace == -1:
            return None
        candidate = _extract_balanced(snippet[brace:])
        if candidate is None:
            return None

    obj = _json_loads_relaxed(candidate)
    if obj is None:
        obj = _parse_loose_action_object(candidate)
    return _normalize_action_object(obj)


def _extract_balanced(s):
    """Return the substring of s that is the first balanced {...} block."""
    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[: i + 1]
    return None


def run_tool(action, auto_yes, mcp_manager=None):
    tool = action["tool"]
    args = action.get("args", {})
    try:
        if tool == "list_dir":
            return list_dir(args)
        if tool == "read_file":
            return read_file(args)
        if tool == "open_file":
            return open_file(args)
        if tool == "write_file":
            return write_file(args, auto_yes)
        if tool == "str_replace":
            return str_replace(args, auto_yes)
        if tool == "run_shell":
            return run_shell(args, auto_yes)
        if tool == "mcp_call":
            if not mcp_manager or not mcp_manager.servers:
                return "ERROR: no MCP servers configured; pass --mcp-config"
            print(f"\n--- about to call MCP tool: {args.get('server')}.{args.get('tool')} ---")
            if not auto_yes and not _confirm():
                return "SKIPPED: user declined mcp_call"
            return mcp_manager.call(args)
        return f"ERROR: unknown tool '{tool}'"
    except KeyError as e:
        return f"ERROR: missing argument {e} for tool {tool}"
    except Exception as e:  # noqa: BLE001 - surface any tool error back to the model
        return f"ERROR: {type(e).__name__}: {e}"


def _normalized_command(text):
    return re.sub(r"\s+", "", text.strip().lower())


def is_exit_command(text):
    return _normalized_command(text) in {"exit()", "退出"}


def is_end_session_command(text):
    normalized = _normalized_command(text)
    exact = {
        "结束",
        "结束任务",
        "结束本次任务",
        "结束当前任务",
        "结束会话",
        "结束本次会话",
        "结束当前会话",
        "结束session",
        "结束当前session",
        "结束本次session",
    }
    if normalized in exact:
        return True
    return (
        "结束" in normalized
        and any(marker in normalized for marker in ("任务", "会话", "session"))
    )


def _looks_like_unverified_completion(reply):
    lowered = reply.lower()
    markers = (
        "i have updated",
        "i updated",
        "i have modified",
        "i modified",
        "i have fixed",
        "i fixed",
        "changes have been made",
        "file has been opened",
        "已修改",
        "改好了",
        "已经修改",
        "已经更新",
        "已经完成",
        "已完成",
    )
    file_words = (
        "file",
        "index.html",
        ".py",
        ".js",
        ".html",
        ".css",
        "文件",
        "代码",
        "修改",
    )
    return any(marker in lowered for marker in markers) and any(word in lowered for word in file_words)


def _is_file_mutation_tool(tool):
    return tool in {"write_file", "str_replace"}


def _tool_succeeded(result):
    return result.startswith("OK:")


def agent_loop(base_url, model, task, messages, auto_yes, max_steps, max_tokens, think,
               stall_timeout=180, show_tokens=True, token_stats=None,
               auto_compact=True, compact_threshold=DEFAULT_COMPACT_THRESHOLD,
               ctx_size=0, memory_file=DEFAULT_MEMORY_FILE,
               compact_keep_messages=DEFAULT_COMPACT_KEEP_MESSAGES,
               compact_max_tokens=DEFAULT_COMPACT_MAX_TOKENS,
               mcp_manager=None):
    messages.append({"role": "user", "content": task})
    token_stats = token_stats or TokenStats(enabled=show_tokens)
    repeats = 0
    last_signature = None
    empty_retried = False
    parse_fails = 0
    unverified_replies = 0
    last_file_mutation_ok = None
    for step in range(1, max_steps + 1):
        print(f"\n=== step {step} ===")
        prompt_est = estimate_message_tokens(base_url, messages) if (show_tokens or auto_compact) else None
        context_limit = ctx_size or DEFAULT_CONTEXT_LIMIT
        if (auto_compact and prompt_est is not None
                and prompt_est >= int(context_limit * compact_threshold)):
            if compact_messages(base_url, model, messages, memory_file,
                                compact_keep_messages, compact_max_tokens, think,
                                stall_timeout):
                prompt_est = estimate_message_tokens(base_url, messages)
        if show_tokens and prompt_est is not None:
            print(f"[tokens input~{prompt_est}; session {token_stats.summary()}]")
        reply, finish_reason, usage = call_llm(base_url, model, messages,
                                               max_tokens=max_tokens, think=think,
                                               stall_timeout=stall_timeout,
                                               show_tokens=show_tokens)
        if show_tokens:
            if usage:
                if usage.get("estimated") and prompt_est is not None:
                    usage["prompt_tokens"] = prompt_est
                    usage["total_tokens"] = prompt_est + int(usage.get("completion_tokens", 0))
                token_stats.add(usage)
                approx = "~" if usage.get("estimated") else ""
                prompt = usage.get("prompt_tokens", 0)
                completion = usage.get("completion_tokens", 0)
                total = usage.get("total_tokens", prompt + completion)
                print(f"[tokens step {approx}prompt={prompt}, completion={completion}, total={total}; "
                      f"session {token_stats.summary()}]")
            elif prompt_est is not None:
                print(f"[tokens step prompt~{prompt_est}; completion unavailable]")

        # The model occasionally returns an empty assistant turn; retry once before
        # giving up so a single stray empty reply does not silently drop the task.
        # Don't append the empty turn to history (it would confuse the next call).
        if not reply.strip():
            if not empty_retried:
                empty_retried = True
                print("[model returned an empty reply, retrying...]")
                continue
            print("[model returned an empty reply again, please retry or rephrase]")
            return
        empty_retried = False

        action = parse_action(reply)
        if action is None:
            if "ACTION" in reply:
                parse_fails += 1
                if parse_fails >= 3:
                    print("\n[could not parse an action 3 times, returning; please rephrase]")
                    return
                # Keep a short placeholder (not the huge broken blob) so history
                # stays small and role alternation is preserved.
                if finish_reason == "length":
                    # The action wasn't malformed JSON -- it was cut off at the token
                    # limit (typically a whole-file write_file). Raise the budget and
                    # steer toward a smaller, surgical edit.
                    max_tokens = min(max_tokens * 2, 16384)
                    print(f"\n[action was cut off at the token limit; raising budget to "
                          f"{max_tokens} and asking for a smaller edit]")
                    messages.append({"role": "assistant",
                                     "content": "[previous action was cut off at the token limit]"})
                    messages.append({"role": "user", "content": (
                        "OBSERVATION:\nERROR: your previous ACTION was cut off because it "
                        "exceeded the output token limit. Do NOT rewrite the whole file in one "
                        "write_file. Make a small, targeted change with str_replace, or write "
                        "the file in several smaller str_replace steps.")})
                    continue
                print("\n[could not parse the ACTION JSON, asking the model to fix it]")
                messages.append({"role": "assistant",
                                 "content": "[previous action was not valid JSON]"})
                messages.append({"role": "user", "content": (
                    "OBSERVATION:\nERROR: your ACTION was not valid JSON and could not "
                    "be parsed. Emit exactly one ACTION followed by a single valid JSON "
                    "object. Escape every backslash in Windows paths as \\\\ (e.g. "
                    '".\\\\WorkDir\\\\hip_gemm.exe"), or just use forward slashes '
                    '("./WorkDir/hip_gemm.exe").')})
                continue
            # No tool call -> treat as a normal chat reply and hand control back
            # to the user instead of looping autonomously (fixes "stuck in loop"
            # on conversational input like "hello").
            if _looks_like_unverified_completion(reply):
                unverified_replies += 1
                if unverified_replies >= 3:
                    print("\n[model kept claiming unverified changes, returning; please rephrase]")
                    return
                print("\n[model claimed file changes without a successful tool observation; asking it to verify]")
                messages.append({"role": "assistant",
                                 "content": "[previous reply claimed unverified file changes]"})
                messages.append({"role": "user", "content": (
                    "OBSERVATION:\nERROR: do not claim files were changed from memory. "
                    "First use read_file to verify the current file, or use a write/edit "
                    "tool and wait for an OK observation. If nothing changed, say so.")})
                continue
            messages.append({"role": "assistant", "content": reply})
            return
        parse_fails = 0
        unverified_replies = 0
        messages.append({"role": "assistant", "content": reply})

        if action["tool"] == "finish":
            answer = action.get("args", {}).get("answer", "")
            if last_file_mutation_ok is not True and _looks_like_unverified_completion(answer):
                unverified_replies += 1
                print("\n[finish claimed file changes without a verified write/edit; asking it to verify]")
                messages.append({"role": "user", "content": (
                    "OBSERVATION:\nERROR: finish answer claims files were changed, but the "
                    "last write/edit tool did not return OK. Use read_file to inspect the "
                    "file or retry the write/edit, then finish only after an OK observation.")})
                if unverified_replies >= 3:
                    print("\n[model kept finishing without verified file changes, returning]")
                    return
                continue
            print("\n=== finished ===")
            print(answer)
            return

        result = run_tool(action, auto_yes, mcp_manager=mcp_manager)
        if _is_file_mutation_tool(action["tool"]):
            last_file_mutation_ok = _tool_succeeded(result)

        # Loop guard: if the model repeats the exact same action and gets the same
        # error several times, stop and return control instead of spinning forever.
        signature = (json.dumps(action, sort_keys=True, ensure_ascii=False), result)
        if result.startswith("ERROR") and signature == last_signature:
            repeats += 1
        else:
            repeats = 0
        last_signature = signature
        if repeats >= 2:
            print("\n[detected a repeated failing action, returning; please adjust the instruction]")
            return

        if len(result) > MAX_OBS_CHARS:
            result = result[:MAX_OBS_CHARS] + "\n... [truncated]"
        # Show command output in full (that's the point of running it); keep a short
        # preview for other tools so the console doesn't flood with file contents.
        if action["tool"] == "run_shell":
            print(f"\n[observation]\n{result}")
        else:
            print(f"\n[observation]\n{result[:500]}{'...' if len(result) > 500 else ''}")
        messages.append({"role": "user", "content": f"OBSERVATION:\n{result}"})

    print("\n[reached max steps, stopping]")


def main():
    parser = argparse.ArgumentParser(description="mini local coding agent")
    parser.add_argument("--base-url", default=None,
                        help="OpenAI-compatible endpoint; default built from --host/--port")
    parser.add_argument("--model", default="local",
                        help="model name (usually ignored by llama-server, any value works)")
    parser.add_argument("--yes", action="store_true",
                        help="skip confirmation before writing files / running commands")
    parser.add_argument("--max-steps", type=int, default=20,
                        help="max steps per task")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="max tokens generated per step (auto-raised if an action is "
                             "truncated; prevents long unresponsive runs)")
    parser.add_argument("--think", action="store_true",
                        help="enable model reasoning/thinking; disabled by default to avoid stalls")
    parser.add_argument("--notes", default=None,
                        help="path to a project conventions file injected into the system prompt "
                             "(default: auto-detect AGENTS.md in the working directory)")
    parser.add_argument("--work-dir", default=".",
                        help="project root for tools, conventions, and .mini-agent memory")
    parser.add_argument("--mcp-config", default=None,
                        help="path to MCP config JSON with mcpServers")
    # llama-server lifecycle.
    parser.add_argument("--no-server", action="store_true",
                        help="do not launch llama-server; connect to an already-running one")
    parser.add_argument("--host", default=DEFAULT_HOST, help="llama-server host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="llama-server port")
    parser.add_argument("--backend", choices=["auto", "vulkan", "hip"], default="auto",
                        help="backend build to auto-detect for llama-server")
    parser.add_argument("--server-bin", default=None,
                        help="path to llama-server executable (default: auto-detect)")
    parser.add_argument("--model-path", default=None,
                        help="path to the .gguf model to serve (default: auto-detect)")
    parser.add_argument("--no-model-prompt", action="store_true",
                        help="auto-select the first detected model instead of prompting")
    parser.add_argument("--ngl", type=int, default=DEFAULT_NGL,
                        help="number of layers to offload to GPU")
    parser.add_argument("--ctx-size", type=int, default=131072,
                        help="context size (-c); 0 leaves the server default")
    parser.add_argument("--server-arg", action="append", default=[], metavar="ARG",
                        help="extra raw argument passed to llama-server (repeatable)")
    parser.add_argument("--server-timeout", type=int, default=300,
                        help="seconds to wait for the server to become ready")
    parser.add_argument("--stall-timeout", type=int, default=180,
                        help="abort a step if the model streams no output for this many seconds")
    parser.add_argument("--tokens", dest="show_tokens", action="store_true", default=True,
                        help="show input token estimates and per-step/session usage")
    parser.add_argument("--no-tokens", dest="show_tokens", action="store_false",
                        help="hide token usage output")
    parser.add_argument("--auto-compact", dest="auto_compact", action="store_true",
                        default=True,
                        help="summarize old history when context usage reaches the threshold")
    parser.add_argument("--no-auto-compact", dest="auto_compact", action="store_false",
                        help="disable automatic long-memory compaction")
    parser.add_argument("--compact-threshold", type=float,
                        default=DEFAULT_COMPACT_THRESHOLD,
                        help="context usage ratio that triggers compaction")
    parser.add_argument("--compact-keep-messages", type=int,
                        default=DEFAULT_COMPACT_KEEP_MESSAGES,
                        help="recent messages to keep verbatim after compaction")
    parser.add_argument("--compact-max-tokens", type=int,
                        default=DEFAULT_COMPACT_MAX_TOKENS,
                        help="max tokens generated for the compaction summary")
    parser.add_argument("--memory-file", default=DEFAULT_MEMORY_FILE,
                        help="markdown file used for compressed session memory")
    args = parser.parse_args()

    work_dir = os.path.abspath(os.path.expanduser(args.work_dir))
    if not os.path.isdir(work_dir):
        print(f"\nError: work directory not found: {work_dir}")
        return
    os.chdir(work_dir)

    base_url = args.base_url or f"http://{args.host}:{args.port}/v1"

    proc = None
    if not args.no_server:
        if server_is_up(base_url):
            print(f"[llama-server already running at {base_url}, reusing it]")
        else:
            try:
                server_bin = resolve_server_bin(args.server_bin, args.backend)
                model_path = resolve_model_path(args.model_path,
                                                prompt=not args.no_model_prompt)
            except RuntimeError as e:
                print(f"\nError: {e}")
                return
            log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.log")
            print(f"[backend] {args.backend}")
            print(f"[server] {server_bin}")
            print(f"[model] {model_path}")
            proc = start_server(server_bin, model_path, args.host, args.port,
                                args.ngl, args.ctx_size, args.server_arg, log_path)
            atexit.register(stop_server, proc)
            try:
                wait_for_server(base_url, proc, timeout=args.server_timeout)
            except RuntimeError as e:
                stop_server(proc)
                print(f"\nError: {e}")
                return
    else:
        if not server_is_up(base_url):
            print(f"[warning] no server reachable at {base_url}; "
                  "start one or drop --no-server")

    mcp_manager = McpManager()
    try:
        mcp_manager.load_config(args.mcp_config)
    except RuntimeError as e:
        print(f"\nError: {e}")
        stop_server(proc)
        return

    print(f"mini-agent -> {base_url} (model={args.model})")
    print(f"[work dir] {os.getcwd()}")
    print('Enter a task and press Enter; use exit() or "退出" to quit.\n')

    base_system_content = f"{SYSTEM_PROMPT}\n\nWorking directory: {os.getcwd()}\nAll relative paths resolve against this directory."
    conv_path, conv_text = load_conventions(args.notes)
    if conv_text:
        base_system_content += (
            f"\n\nProject conventions (from {os.path.basename(conv_path)}; "
            f"follow these strictly):\n{conv_text}"
        )
        print(f"[loaded conventions from {conv_path}]")
    mcp_tools = mcp_manager.describe_tools()
    if mcp_tools:
        base_system_content += (
            "\n\nConfigured MCP tools. Call them via mcp_call with server, tool, and args:\n"
            f"{mcp_tools}"
        )
        print("[loaded MCP tools]")

    def new_session_messages(print_loaded=False):
        system_content = base_system_content
        memory_text = load_memory(args.memory_file) if args.auto_compact else None
        if memory_text:
            system_content += (
                f"\n\nPersistent session memory (from {args.memory_file}; use it as "
                f"compressed prior context):\n{memory_text}"
            )
            if print_loaded:
                print(f"[loaded memory from {args.memory_file}]")
        return [{"role": "system", "content": system_content}]

    messages = new_session_messages(print_loaded=True)
    token_stats = TokenStats(enabled=args.show_tokens)
    try:
        while True:
            try:
                task = input("task> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nbye")
                break
            if not task:
                continue
            if is_exit_command(task):
                print("bye")
                break
            if is_end_session_command(task):
                if len(messages) > 1 and args.auto_compact:
                    try:
                        compact_messages(base_url, args.model, messages, args.memory_file,
                                         keep_messages=0,
                                         max_tokens=args.compact_max_tokens,
                                         think=args.think,
                                         stall_timeout=args.stall_timeout)
                    except RuntimeError as e:
                        print(f"[session memory save failed: {e}]")
                messages = new_session_messages(print_loaded=True)
                token_stats = TokenStats(enabled=args.show_tokens)
                print("[session ended; waiting for a new task]")
                continue
            try:
                agent_loop(base_url, args.model, task, messages, args.yes,
                           args.max_steps, args.max_tokens, args.think,
                           stall_timeout=args.stall_timeout,
                           show_tokens=args.show_tokens,
                           token_stats=token_stats,
                           auto_compact=args.auto_compact,
                           compact_threshold=args.compact_threshold,
                           ctx_size=args.ctx_size,
                           memory_file=args.memory_file,
                           compact_keep_messages=args.compact_keep_messages,
                           compact_max_tokens=args.compact_max_tokens,
                           mcp_manager=mcp_manager)
            except RuntimeError as e:
                print(f"\nError: {e}")
            except KeyboardInterrupt:
                print("\n[current task interrupted]")
    finally:
        mcp_manager.stop()
        stop_server(proc)


if __name__ == "__main__":
    main()
