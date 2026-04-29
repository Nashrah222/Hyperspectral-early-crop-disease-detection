"""Runner script that captures all output cleanly to a log file."""
import subprocess
import sys

result = subprocess.run(
    [sys.executable, "main.py"],
    cwd=r"d:\Projects\crop_disease\crop_disease",
    capture_output=True,
    text=True,
    encoding="utf-8",
    errors="replace",
    env={
        **__import__("os").environ,
        "PYTHONIOENCODING": "utf-8",
        "TF_CPP_MIN_LOG_LEVEL": "2",
    },
)

with open("run_log_clean.txt", "w", encoding="utf-8") as f:
    f.write("=== STDOUT ===\n")
    f.write(result.stdout)
    f.write("\n\n=== STDERR ===\n")
    f.write(result.stderr)
    f.write(f"\n\n=== EXIT CODE: {result.returncode} ===\n")

print(f"Exit code: {result.returncode}")
print("Log written to run_log_clean.txt")
