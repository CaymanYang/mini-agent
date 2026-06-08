#!/usr/bin/env python3
"""mini local agent: a tiny ReAct coding agent for an OpenAI-compatible local server (llama.cpp)."""

import argparse
import atexit
import json
import os
import queue
import re
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
DEFAULT_SERVER_BIN = r"C:\Users\bookery\llama.cpp\build-x64-windows-vulkan-release\bin\llama-server.exe"
DEFAULT_MODEL_PATH = r"C:\Users\bookery\llama.cpp\mymodels\gemma-4-12b-it-GGUF\gemma-4-12b-it-Q4_K_M.gguf"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_NGL = 99

SYSTEM_PROMPT = """You are a minimal coding agent. You solve the user's task by calling tools, one step at a time.

To call a tool, output a line containing exactly ACTION, then a single JSON object on the following lines:
ACTION
{"tool": "<name>", "args": { ... }}

Available tools:
- list_dir: {"path": str (optional, default ".")}  -> list files/dirs at a path
- read_file: {"path": str}  -> returns file content with line numbers
- str_replace: {"path": str, "old_string": str, "new_string": str}  -> EDIT an existing file by replacing one exact, unique snippet
- write_file: {"path": str, "content": str, "overwrite": bool (optional)}  -> create a NEW file; refuses to overwrite an existing file unless overwrite=true
- run_shell: {"command": str}  -> runs a shell command, returns stdout/stderr/exit code
- finish: {"answer": str}  -> end the task and give the final answer to the user

Editing files (IMPORTANT):
- To change an EXISTING file you MUST use str_replace, never write_file. write_file is only for creating a brand-new file.
- Workflow: read_file first, then str_replace with old_string copied EXACTLY from that file content (drop the "   N|" line-number prefix), including enough surrounding lines to be unique.
- Make several small str_replace edits instead of rewriting the whole file.

Example of editing an existing file:
ACTION
{"tool": "read_file", "args": {"path": "main.c"}}
(observation shows: "    10|    int n = 5;")
ACTION
{"tool": "str_replace", "args": {"path": "main.c", "old_string": "    int n = 5;", "new_string": "    int n = 10;"}}

Rules:
- Emit exactly ONE ACTION per reply. Do not emit more than one JSON object.
- The JSON must be valid. Put file contents inside the string values (escape newlines as \\n).
- Relative paths resolve against the working directory shown below; you usually do not need absolute paths. Use list_dir to discover files.
- After each ACTION you will receive an OBSERVATION with the result. Use it to decide the next step.
- Keep reasoning brief. When the task is done, call finish."""


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


def call_llm(base_url, model, messages, max_tokens=2048, think=False, stream=True,
             stop_on_action=True, stall_timeout=180):
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
        text = choice["message"]["content"]
        print(text, end="", flush=True)
        print()
        return text, choice.get("finish_reason")

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
    last_data = time.time()
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
            choice = obj.get("choices", [{}])[0]
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
            delta = choice.get("delta", {})
            # Reasoning tokens arrive separately; show them so output never looks frozen.
            reasoning = delta.get("reasoning_content")
            if reasoning:
                if not in_reasoning:
                    print("[thinking] ", end="", flush=True)
                    in_reasoning = True
                print(reasoning, end="", flush=True)
            piece = delta.get("content")
            if piece:
                if in_reasoning:
                    print("\n", end="", flush=True)
                    in_reasoning = False
                parts.append(piece)
                print(piece, end="", flush=True)
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
    return "".join(parts), finish_reason


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


def write_file(args, auto_yes):
    path = args["path"]
    content = args.get("content", "")
    if os.path.exists(path) and not args.get("overwrite"):
        # Steer the model toward editing instead of rewriting existing files.
        return (f"ERROR: {path} already exists. To EDIT an existing file use "
                "str_replace. To replace the entire file anyway, pass "
                '"overwrite": true.')
    print(f"\n--- about to write file: {path} ({len(content)} chars) ---")
    if not auto_yes and not _confirm():
        return "SKIPPED: user declined write_file"
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"OK: wrote {len(content)} chars to {path}"


def str_replace(args, auto_yes):
    path = args["path"]
    old = args["old_string"]
    new = args.get("new_string", "")
    if not os.path.exists(path):
        return f"ERROR: file not found: {path}"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    count = content.count(old)
    if count == 0:
        return "ERROR: old_string not found. Read the file again to copy exact text."
    if count > 1:
        return (f"ERROR: old_string matched {count} times; it must be unique. "
                "Include more surrounding context.")
    print(f"\n--- about to edit file: {path} (1 replacement, {len(old)}->{len(new)} chars) ---")
    if not auto_yes and not _confirm():
        return "SKIPPED: user declined str_replace"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content.replace(old, new))
    return f"OK: replaced 1 occurrence in {path}"


def _windows_shell():
    # Prefer PowerShell 7 (pwsh) which supports && / ||; fall back to Windows
    # PowerShell 5.1 (powershell) which does not.
    return "pwsh" if shutil.which("pwsh") else "powershell"


def run_shell(args, auto_yes):
    command = args["command"]
    print(f"\n--- about to run command: {command} ---")
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


def _confirm():
    try:
        ans = input("Confirm? [y/N] ").strip().lower()
    except EOFError:
        return False
    return ans in ("y", "yes")


def parse_action(text):
    """Extract the JSON action object following an ACTION marker (or the first JSON object)."""
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

    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        # Models frequently emit raw Windows paths like ".\WorkDir\app.exe".
        # Backslashes such as \W or \h are invalid JSON escapes, so escape any
        # backslash that isn't part of a valid escape and try once more.
        try:
            obj = json.loads(re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", candidate))
        except json.JSONDecodeError:
            return None
    if isinstance(obj, dict) and "tool" in obj:
        args = obj.get("args")
        if not isinstance(args, dict):
            # Some models flatten args to the top level instead of nesting them
            # under "args" (e.g. {"tool":"run_shell","command":"ls"}). Accept that.
            args = {k: v for k, v in obj.items() if k != "tool"}
        obj["args"] = args
        return obj
    return None


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


def run_tool(action, auto_yes):
    tool = action["tool"]
    args = action.get("args", {})
    try:
        if tool == "list_dir":
            return list_dir(args)
        if tool == "read_file":
            return read_file(args)
        if tool == "write_file":
            return write_file(args, auto_yes)
        if tool == "str_replace":
            return str_replace(args, auto_yes)
        if tool == "run_shell":
            return run_shell(args, auto_yes)
        return f"ERROR: unknown tool '{tool}'"
    except KeyError as e:
        return f"ERROR: missing argument {e} for tool {tool}"
    except Exception as e:  # noqa: BLE001 - surface any tool error back to the model
        return f"ERROR: {type(e).__name__}: {e}"


def agent_loop(base_url, model, task, messages, auto_yes, max_steps, max_tokens, think,
               stall_timeout=180):
    messages.append({"role": "user", "content": task})
    repeats = 0
    last_signature = None
    empty_retried = False
    parse_fails = 0
    for step in range(1, max_steps + 1):
        print(f"\n=== step {step} ===")
        reply, finish_reason = call_llm(base_url, model, messages,
                                        max_tokens=max_tokens, think=think,
                                        stall_timeout=stall_timeout)

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
            messages.append({"role": "assistant", "content": reply})
            return
        parse_fails = 0
        messages.append({"role": "assistant", "content": reply})

        if action["tool"] == "finish":
            answer = action.get("args", {}).get("answer", "")
            print("\n=== finished ===")
            print(answer)
            return

        result = run_tool(action, auto_yes)

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
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="max tokens generated per step (auto-raised if an action is "
                             "truncated; prevents long unresponsive runs)")
    parser.add_argument("--think", action="store_true",
                        help="enable model reasoning/thinking; disabled by default to avoid stalls")
    parser.add_argument("--notes", default=None,
                        help="path to a project conventions file injected into the system prompt "
                             "(default: auto-detect AGENTS.md in the working directory)")
    # llama-server lifecycle.
    parser.add_argument("--no-server", action="store_true",
                        help="do not launch llama-server; connect to an already-running one")
    parser.add_argument("--host", default=DEFAULT_HOST, help="llama-server host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="llama-server port")
    parser.add_argument("--server-bin", default=DEFAULT_SERVER_BIN,
                        help="path to llama-server executable")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH,
                        help="path to the .gguf model to serve")
    parser.add_argument("--ngl", type=int, default=DEFAULT_NGL,
                        help="number of layers to offload to GPU")
    parser.add_argument("--ctx-size", type=int, default=0,
                        help="context size (-c); 0 leaves the server default")
    parser.add_argument("--server-arg", action="append", default=[], metavar="ARG",
                        help="extra raw argument passed to llama-server (repeatable)")
    parser.add_argument("--server-timeout", type=int, default=300,
                        help="seconds to wait for the server to become ready")
    parser.add_argument("--stall-timeout", type=int, default=180,
                        help="abort a step if the model streams no output for this many seconds")
    args = parser.parse_args()

    base_url = args.base_url or f"http://{args.host}:{args.port}/v1"

    proc = None
    if not args.no_server:
        if server_is_up(base_url):
            print(f"[llama-server already running at {base_url}, reusing it]")
        else:
            log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.log")
            proc = start_server(args.server_bin, args.model_path, args.host, args.port,
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

    print(f"mini-agent -> {base_url} (model={args.model})")
    print("Enter a task and press Enter; empty line or Ctrl+C to quit.\n")

    system_content = f"{SYSTEM_PROMPT}\n\nWorking directory: {os.getcwd()}\nAll relative paths resolve against this directory."
    conv_path, conv_text = load_conventions(args.notes)
    if conv_text:
        system_content += (
            f"\n\nProject conventions (from {os.path.basename(conv_path)}; "
            f"follow these strictly):\n{conv_text}"
        )
        print(f"[loaded conventions from {conv_path}]")
    messages = [{"role": "system", "content": system_content}]
    try:
        while True:
            try:
                task = input("task> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nbye")
                break
            if not task:
                print("bye")
                break
            try:
                agent_loop(base_url, args.model, task, messages, args.yes,
                           args.max_steps, args.max_tokens, args.think,
                           stall_timeout=args.stall_timeout)
            except RuntimeError as e:
                print(f"\nError: {e}")
            except KeyboardInterrupt:
                print("\n[current task interrupted]")
    finally:
        stop_server(proc)


if __name__ == "__main__":
    main()
