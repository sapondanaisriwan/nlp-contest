set windows-shell := ["powershell.exe", "-NoProfile", "-Command"]

# Show available commands
default:
    @just --list

# Start a NEW 277-question development run (CPU Q8)
run:
    @& .venv/Scripts/python.exe -X utf8 scripts/local_tasks.py run

# Show progress of the latest run
status:
    @& .venv/Scripts/python.exe -X utf8 scripts/local_tasks.py status

# Follow the latest log; Ctrl+C stops watching, not the existing run
logs:
    @& .venv/Scripts/python.exe -X utf8 scripts/local_tasks.py logs

# Show final score and output paths
score:
    @& .venv/Scripts/python.exe -X utf8 scripts/local_tasks.py score

# Check the latest precision48 trial (does not start inference)
precision-status:
    @& .venv/Scripts/python.exe -X utf8 scripts/local_tasks.py precision-status

# Follow precision48 progress
precision-logs:
    @& .venv/Scripts/python.exe -X utf8 scripts/local_tasks.py precision-logs

# Read the precision48 score
precision-score:
    @& .venv/Scripts/python.exe -X utf8 scripts/local_tasks.py precision-score

# NEW: targeted evidence and short-answer instructions on the SAME 48 dev questions
improve48:
    @& .venv/Scripts/python.exe -X utf8 -u scripts/trial_precision.py --strategy focused_v5 --prompt targeted --label improve

# Full development evaluation, excluding holdout
improve277:
    @& .venv/Scripts/python.exe -X utf8 -u scripts/trial_precision.py --strategy focused_v5 --prompt targeted --label improve --all-dev

# Optional ablation: new evidence only, old prompt
improve48-evidence:
    @& .venv/Scripts/python.exe -X utf8 -u scripts/trial_precision.py --strategy focused_v4 --prompt basic --label improve

improve-status:
    @& .venv/Scripts/python.exe -X utf8 scripts/local_tasks.py improve-status

improve-logs:
    @& .venv/Scripts/python.exe -X utf8 scripts/local_tasks.py improve-logs

improve-score:
    @& .venv/Scripts/python.exe -X utf8 scripts/local_tasks.py improve-score
