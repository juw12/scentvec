#!/usr/bin/env python3
"""Run a command fully detached from the calling terminal.

    python3 scripts/detach.py <log-file> <command> [args...]

The command gets its own session (double fork + setsid), so it keeps running
when the launching shell or terminal goes away.  Output goes to the log file
and the exit status is appended as a final "exit=<code>" line.
"""
import os
import subprocess
import sys


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    log_path = sys.argv[1]
    command = sys.argv[2:]
    if os.fork() > 0:
        return
    os.setsid()
    if os.fork() > 0:
        os._exit(0)
    with open(os.devnull, "rb") as null_input, open(log_path, "ab", buffering=0) as log:
        status = subprocess.call(command, stdin=null_input, stdout=log, stderr=log)
        log.write(("exit=%d\n" % status).encode())
    os._exit(0)


if __name__ == "__main__":
    main()
