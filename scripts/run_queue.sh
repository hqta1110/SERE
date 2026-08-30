#!/usr/bin/env bash
# Priority-ordered driver for the unattended queue.
#
# Single script rather than four commands typed into a tmux window: buffered
# stdin in a tmux pane has twice caused a job to fire on top of a running one
# and destroyed cells. Here the ordering is explicit and nothing is buffered.
#
# ORDER IS BY IMPORTANCE TO ARES, most important first:
#
# 1. validate_ares.sh   GATING. Four ways the published numbers could be
#    artifacts. Test 1 in particular (PYTHONPATH sets an NCCL repair that
#    baselines never get) would mean EVERY ARES-vs-baseline ratio is
#    contaminated by an environment difference rather than by expert skipping.
#    Nothing else is worth interpreting until this is known, and it also yields
#    the multiplicative correction factor to apply to jobs 2-4 if it fails.
#    Test 3 (accuracy noise floor from two identical runs) similarly gates
#    whether a -3.33 point delta is a finding at all.
#
# 2. computebound_ab.sh The only queued experiment that could produce a
#    POSITIVE headline. Every number so far is memory-bound decode -- the regime
#    where masking structurally cannot win, because masked weights stay resident
#    (measured: same 5.82 GiB KV cache as baseline). Arithmetic is the one axis
#    where ARES leads: 3.60 of 8 slots vs 8 of 8 for both SERE and REAP.
#
# 3. math500_frontier.sh Second quality benchmark. The entire quality axis
#    currently rests on GSM8K strict-match alone, which is thin for ICML/ICLR
#    and not for lack of trying (HumanEval was measured and proved unusable).
#
# 4. safe_concurrency.sh Lowest value and expected to make ARES look worse: the
#    published concurrency table used frac 0.20, past the accuracy cliff. Worth
#    fixing because it is not a defensible comparison, but it is a correction,
#    not a discovery.
#
# Every job is independently resumable (per-cell artifact checks), so an
# interruption anywhere leaves complete rows rather than a smear.
set -uo pipefail

SV=/home/PC/SERE_v1
AR=/home/PC/new-efficient-moe
LOG=$SV/logs

echo "=== QUEUE START $(date '+%F %T') ==="

# Wait for the Qwen3.6 sweep to release the GPUs (it holds this lock for its
# lifetime). `flock FILE -c true` acquires and immediately releases -- do NOT
# hold the fd, or child jobs taking the same lock would deadlock against us.
if pgrep -f 'q36_sweep.sh' >/dev/null 2>&1; then
  echo "[queue] waiting for q36_sweep to finish..."
  flock /tmp/q36_sweep.lock -c true
fi
bash "$AR/scripts/drain_gpus.sh" >/dev/null 2>&1

run() {  # run <n> <label> <script> <logfile>
  echo ""
  echo "############################################################"
  echo "[queue] $1/4  $2   $(date '+%F %T')"
  echo "############################################################"
  bash "$SV/scripts/$3" 2>&1 | tee "$LOG/$4"
  bash "$AR/scripts/drain_gpus.sh" >/dev/null 2>&1
  sleep 5
}

run 1 "ADVERSARIAL VALIDATION (gating)"  validate_ares.sh      validate.log
run 2 "COMPUTE-BOUND A/B"                computebound_ab.sh    computebound.log
run 3 "MATH500 FRONTIER"                 math500_frontier.sh   math500.log
run 4 "QUALITY-SAFE CONCURRENCY"         safe_concurrency.sh   safe_conc.log

echo ""
echo "=== QUEUE EXHAUSTED $(date '+%F %T') ==="
echo "Read in this order:"
echo "  1. logs/validate.log     -- tail it; PASS/FAIL verdict is at the bottom"
echo "  2. logs/computebound.log -- the x(total) column across 3 traffic shapes"
echo "  3. logs/math500.log      -- second quality benchmark vs the GSM8K frontier"
echo "  4. logs/safe_conc.log    -- honest concurrency at deployable settings"
echo ""
echo "If validate.log TEST 1 FAILED, jobs 2-4 baselines need the env_off"
echo "correction applied -- the factor is printed in validate.log. No rerun"
echo "needed, it is a multiplicative correction."
