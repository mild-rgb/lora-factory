#!/bin/sh
# While r8x is paused: wait for a running GPU job (by PID) to exit, run one more job, then resume r8x.
# r8x resumes even if the extra job fails.  Usage: sh after_job.sh <pid> <job script>
L=/content/r8x/slot.log
echo "$(date -u +%T) after_job: waiting for pid $1" >> $L
while kill -0 "$1" 2>/dev/null; do sleep 20; done
echo "$(date -u +%T) running $2" >> $L
sh "$2" >> $L 2>&1
echo "$(date -u +%T) job exited with $?; resuming r8x" >> $L
python /content/r8x_code/resume_r8x.py >> $L 2>&1
