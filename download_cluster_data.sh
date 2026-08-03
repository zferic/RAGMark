#!/bin/bash
# Downloads data needed for a local run from the discovery cluster.
# Edit the LOCAL_* paths below, then run: bash download_cluster_data.sh

set -e

REMOTE_USER="feric.z"
REMOTE_HOST="xfer.discovery.neu.edu"
REMOTE="$REMOTE_USER@$REMOTE_HOST"
SSH_MUX="/tmp/ssh_mux_discovery"

# -------------------------------------------------------------------------
# Edit these to match where you want files locally
# (also update newsrc/config.py to match after downloading)
# -------------------------------------------------------------------------
LOCAL_INDEX_DIR="/media/zman/extrahd2/data/wikiindex2"
LOCAL_CORPUS_DIR="/media/zman/extrahd2/data/flashrag"
LOCAL_DATASET_DIR="/media/zman/extrahd2/data/flashrag/dataset"
# -------------------------------------------------------------------------

echo "=== Opening SSH connection (enter password once) ==="
ssh -M -S "$SSH_MUX" -fN -o ControlPersist=600 "$REMOTE"

rsync_via_mux() {
    rsync -avz --progress -e "ssh -S $SSH_MUX" "$@"
}

# 1. FAISS index
echo ""
echo "=== Downloading FAISS index ==="
mkdir -p "$LOCAL_INDEX_DIR"
rsync_via_mux \
    "$REMOTE:/projects/nucar/feric.z/wikiindex2/e5-base-v2-2018/" \
    "$LOCAL_INDEX_DIR/e5-base-v2-2018/"

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
ssh -S "$SSH_MUX" -O exit "$REMOTE" 2>/dev/null || true

echo ""
echo "Done. Update newsrc/config.py with:"
echo "  WIKI_INDEX_DIR = \"$LOCAL_INDEX_DIR/\""
echo "  WIKI_CORPUS_2018 = \"$LOCAL_CORPUS_DIR/test_sample_2018_sent.jsonl\""
echo "  DATASET_DIR    = \"$LOCAL_DATASET_DIR\""
