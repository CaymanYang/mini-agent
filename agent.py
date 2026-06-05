#!/usr/bin/env python3
"""mini local agent: a tiny ReAct coding agent for an OpenAI-compatible local server (llama.cpp)."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
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


def call_llm(base_url, model, messages, max_tokens=2048, think=False, stream=True,
             stop_on_action=True):
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
        resp = urllib.request.urlopen(req)
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to connect to local model server {url}: {e}")

    if not stream:
        body = json.loads(resp.read().decode("utf-8"))
        text = body["choices"][0]["message"]["content"]
        print(text, end="", flush=True)
        print()
        return text

    parts = []
    in_reasoning = False
    for raw in resp:
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
        delta = obj.get("choices", [{}])[0].get("delta", {})
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
            # Early stop: once a full ACTION is available, don't wait for the model
            # to keep rambling (it often appends extra actions/text up to max_tokens).
            if stop_on_action and "}" in piece:
                joined = "".join(parts)
                if "ACTION" in joined and parse_action(joined) is not None:
                    break
    try:
        resp.close()
    except Exception:
        pass
    print()
    return "".join(parts)


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


def agent_loop(base_url, model, task, messages, auto_yes, max_steps, max_tokens, think):
    messages.append({"role": "user", "content": task})
    repeats = 0
    last_signature = None
    empty_retried = False
    for step in range(1, max_steps + 1):
        print(f"\n=== step {step} ===")
        reply = call_llm(base_url, model, messages, max_tokens=max_tokens, think=think)

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
        messages.append({"role": "assistant", "content": reply})

        action = parse_action(reply)
        if action is None:
            # No tool call -> treat as a normal chat reply and hand control back
            # to the user instead of looping autonomously (fixes "stuck in loop"
            # on conversational input like "hello").
            return

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
        print(f"\n[observation]\n{result[:500]}{'...' if len(result) > 500 else ''}")
        messages.append({"role": "user", "content": f"OBSERVATION:\n{result}"})

    print("\n[reached max steps, stopping]")


def main():
    parser = argparse.ArgumentParser(description="mini local coding agent")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1",
                        help="OpenAI-compatible endpoint (default: llama-server)")
    parser.add_argument("--model", default="local",
                        help="model name (usually ignored by llama-server, any value works)")
    parser.add_argument("--yes", action="store_true",
                        help="skip confirmation before writing files / running commands")
    parser.add_argument("--max-steps", type=int, default=20,
                        help="max steps per task")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="max tokens generated per step (prevents long unresponsive runs)")
    parser.add_argument("--think", action="store_true",
                        help="enable model reasoning/thinking; disabled by default to avoid stalls")
    args = parser.parse_args()

    print(f"mini-agent -> {args.base_url} (model={args.model})")
    print("Enter a task and press Enter; empty line or Ctrl+C to quit.\n")

    system_content = f"{SYSTEM_PROMPT}\n\nWorking directory: {os.getcwd()}\nAll relative paths resolve against this directory."
    messages = [{"role": "system", "content": system_content}]
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
            agent_loop(args.base_url, args.model, task, messages, args.yes,
                       args.max_steps, args.max_tokens, args.think)
        except RuntimeError as e:
            print(f"\nError: {e}")
        except KeyboardInterrupt:
            print("\n[current task interrupted]")


if __name__ == "__main__":
    main()
