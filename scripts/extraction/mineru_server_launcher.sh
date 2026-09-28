#!/usr/bin/env bash

MINERU_SERVER_PID=""
MINERU_SERVER_STARTED=0
MINERU_TERM_REQUESTED=0

mineru_timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

mineru_python() {
  if [[ -n "${MINERU_PYTHON:-}" ]]; then
    printf '%s\n' "$MINERU_PYTHON"
  elif command -v python3 >/dev/null 2>&1; then
    command -v python3
  elif command -v python >/dev/null 2>&1; then
    command -v python
  else
    return 1
  fi
}

mineru_check_pdftext_compatibility() {
  local python_bin="$1"

  "$python_bin" - <<'PY'
from importlib.metadata import PackageNotFoundError, version

expected = "0.6.3"
try:
    installed = version("pdftext")
except PackageNotFoundError as exc:
    raise SystemExit("MinerU runtime dependency is missing: pdftext") from exc

if installed != expected:
    raise SystemExit(
        f"Incompatible pdftext version {installed}; MinerU 3.4.0 requires {expected} "
        "to avoid the non-iterable PageChars API"
    )
PY
}

mineru_health_check() {
  local url="$1"
  local python_bin
  python_bin="$(mineru_python)" || return 1
  "$python_bin" - "$url" <<'PY'
import json
import sys
import urllib.error
import urllib.request

url = sys.argv[1].rstrip("/") + "/health"
try:
    with urllib.request.urlopen(url, timeout=2) as response:
        if response.status < 200 or response.status >= 300:
            sys.exit(1)
        body = response.read()
except (OSError, urllib.error.URLError, TimeoutError):
    sys.exit(1)

try:
    json.loads(body.decode("utf-8"))
except (UnicodeDecodeError, json.JSONDecodeError):
    sys.exit(1)
PY
}

mineru_start_server() {
  local server_cmd="$1"
  local server_log="$2"

  mkdir -p "$(dirname "$server_log")"
  printf '[%s] Starting MinerU server: %s\n' "$(mineru_timestamp)" "$server_cmd" >>"$server_log"

  if command -v setsid >/dev/null 2>&1; then
    PYTHONUNBUFFERED=1 setsid bash -lc "exec $server_cmd" >>"$server_log" 2>&1 &
  else
    PYTHONUNBUFFERED=1 bash -lc "exec $server_cmd" >>"$server_log" 2>&1 &
  fi
  MINERU_SERVER_PID=$!
  MINERU_SERVER_STARTED=1
}

mineru_stop_server() {
  local server_log="$1"

  if [[ -z "${MINERU_SERVER_PID:-}" ]] || ! kill -0 "$MINERU_SERVER_PID" >/dev/null 2>&1; then
    return 0
  fi

  printf '[%s] Stopping MinerU server pid=%s\n' "$(mineru_timestamp)" "$MINERU_SERVER_PID" >>"$server_log"
  if command -v setsid >/dev/null 2>&1; then
    kill -TERM "-$MINERU_SERVER_PID" >/dev/null 2>&1 || kill -TERM "$MINERU_SERVER_PID" >/dev/null 2>&1 || true
  else
    kill -TERM "$MINERU_SERVER_PID" >/dev/null 2>&1 || true
  fi

  local deadline=$((SECONDS + ${MINERU_SHUTDOWN_TIMEOUT:-20}))
  while kill -0 "$MINERU_SERVER_PID" >/dev/null 2>&1 && (( SECONDS < deadline )); do
    sleep 1
  done

  if kill -0 "$MINERU_SERVER_PID" >/dev/null 2>&1; then
    printf '[%s] MinerU server did not exit after SIGTERM; sending SIGKILL\n' "$(mineru_timestamp)" >>"$server_log"
    if command -v setsid >/dev/null 2>&1; then
      kill -KILL "-$MINERU_SERVER_PID" >/dev/null 2>&1 || kill -KILL "$MINERU_SERVER_PID" >/dev/null 2>&1 || true
    else
      kill -KILL "$MINERU_SERVER_PID" >/dev/null 2>&1 || true
    fi
  fi

  wait "$MINERU_SERVER_PID" >/dev/null 2>&1 || true
}

mineru_wait_for_ready() {
  local api_url="$1"
  local server_log="$2"
  local timeout_seconds="$3"

  local deadline=$((SECONDS + timeout_seconds))
  while (( SECONDS < deadline )); do
    if ! kill -0 "$MINERU_SERVER_PID" >/dev/null 2>&1; then
      local server_status=0
      wait "$MINERU_SERVER_PID" || server_status=$?
      printf '[%s] MinerU server exited before readiness with status %s\n' \
        "$(mineru_timestamp)" "$server_status" >>"$server_log"
      printf 'MinerU server exited before readiness with status %s. See %s\n' \
        "$server_status" "$server_log" >&2
      return 1
    fi
    if mineru_health_check "$api_url"; then
      printf '[%s] MinerU server is ready at %s\n' "$(mineru_timestamp)" "$api_url" >>"$server_log"
      return 0
    fi
    sleep 2
  done

  printf 'Timed out after %s seconds waiting for MinerU readiness at %s. See %s\n' \
    "$timeout_seconds" "$api_url" "$server_log" >&2
  return 1
}

mineru_run_caller_with_server() {
  local api_url="$1"
  local server_cmd="$2"
  local server_log="$3"
  local caller_log="$4"
  local startup_timeout="$5"
  shift 5
  local caller_cmd=("$@")

  local caller_status=0
  mkdir -p "$(dirname "$caller_log")"

  on_term() {
    MINERU_TERM_REQUESTED=1
    printf '[%s] Received SIGTERM; cleaning up MinerU server\n' "$(mineru_timestamp)" >>"$server_log"
    mineru_stop_server "$server_log"
    exit 143
  }
  trap on_term TERM INT

  mineru_start_server "$server_cmd" "$server_log"
  if ! mineru_wait_for_ready "$api_url" "$server_log" "$startup_timeout"; then
    mineru_stop_server "$server_log"
    return 124
  fi

  {
    printf '[%s] Starting caller: %q' "$(mineru_timestamp)" "${caller_cmd[0]}"
    printf ' %q' "${caller_cmd[@]:1}"
    printf '\n'
  } | tee -a "$caller_log"

  set +e
  PYTHONUNBUFFERED=1 "${caller_cmd[@]}" 2>&1 | tee -a "$caller_log"
  caller_status=${PIPESTATUS[0]}
  set -e

  trap - TERM INT
  mineru_stop_server "$server_log"
  return "$caller_status"
}
