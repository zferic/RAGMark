#!/bin/bash
# Downloads data needed for a local run from the discovery cluster.
# Edit the LOCAL_* paths below, then run: bash download_cluster_data.sh

set -euo pipefail

REMOTE_USER="feric.z"
REMOTE_HOST="xfer.discovery.neu.edu"
REMOTE="$REMOTE_USER@$REMOTE_HOST"
SSH_MUX="/tmp/ssh_mux_discovery"

# -------------------------------------------------------------------------
# Edit these to match where you want files locally
# (also update config.py to match after downloading)
# -------------------------------------------------------------------------
LOCAL_INDEX_DIR="/mnt/nvme1n1p1/zman/ragmark_data/wikiindex"
LOCAL_CORPUS_DIR="/mnt/nvme1n1p1/zman/ragmark_data/corpus"
LOCAL_DATASET_DIR="/mnt/nvme1n1p1/zman/ragmark_data/datasets"
# -------------------------------------------------------------------------

echo "=== Opening SSH connection (enter password once) ==="

cleanup_mux() {
    ssh -S "$SSH_MUX" -O exit "$REMOTE" >/dev/null 2>&1 || true
    rm -f "$SSH_MUX"
}

if [[ -e "$SSH_MUX" ]]; then
    if [[ -S "$SSH_MUX" ]] && ssh -S "$SSH_MUX" -O check "$REMOTE" >/dev/null 2>&1; then
        echo "Reusing existing SSH control connection."
    else
        echo "Removing stale SSH mux path: $SSH_MUX"
        rm -f "$SSH_MUX"
    fi
fi

trap cleanup_mux EXIT INT TERM

if ! ssh -S "$SSH_MUX" -O check "$REMOTE" >/dev/null 2>&1; then
    ssh -M -S "$SSH_MUX" -fN \
        -o ControlMaster=auto \
        -o ControlPersist=600 \
        -o ServerAliveInterval=30 \
        -o ServerAliveCountMax=5 \
        "$REMOTE"
fi

rsync_via_mux() {
    rsync -av --human-readable --info=progress2 --partial --inplace --append-verify \
    --rsync-path="/usr/bin/rsync" \
    -e "ssh -T -S $SSH_MUX" "$@"
}

# 1. FAISS index
echo ""
echo "=== Downloading FAISS index ==="
mkdir -p "$LOCAL_INDEX_DIR/e5-base-v2-2018/index"

# Clean up a bad artifact from earlier wildcard-based attempts, if present.
if [[ -f "$LOCAL_INDEX_DIR/e5-base-v2-2018/index/*" ]]; then
    rm -f "$LOCAL_INDEX_DIR/e5-base-v2-2018/index/*"
fi

rsync_via_mux \
    "$REMOTE:/projects/nucar/feric.z/wikiindex2/e5-base-v2-2018/index/e5_Flat.index" \
    "$LOCAL_INDEX_DIR/e5-base-v2-2018/index/e5_Flat.index"

# 2. Wikipedia 2018 corpus
echo ""
echo "=== Downloading Wikipedia 2018 corpus ==="
mkdir -p "$LOCAL_CORPUS_DIR"
rsync_via_mux \
    "$REMOTE:/projects/nucar/feric.z/FLashRAG-z/test_sample_2018_sent.jsonl" \
    "$LOCAL_CORPUS_DIR/test_sample_2018_sent.jsonl"

# 3. Eval datasets
echo ""
echo "=== Downloading eval datasets ==="
mkdir -p "$LOCAL_DATASET_DIR"
for ds in hotpotqa nq squad triviaqa popqa webquestions; do
    rsync_via_mux \
        "$REMOTE:/home/feric.z/FlashRAG/examples/quick_start/dataset/${ds}_dataset.jsonl" \
        "$LOCAL_DATASET_DIR/${ds}_dataset.jsonl"
done

echo ""
echo "=== Closing SSH connection ==="
cleanup_mux

echo ""
echo "Done. Update config.py with:"
echo "  WIKI_INDEX_DIR = \"$LOCAL_INDEX_DIR/\""
echo "  WIKI_CORPUS_2018 = \"$LOCAL_CORPUS_DIR/test_sample_2018_sent.jsonl\""
echo "  DATASET_DIR    = \"$LOCAL_DATASET_DIR\""
