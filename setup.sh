#!/bin/bash
# Provision the preprocessed MIDI pickles that train.py reads through
# `--pickle_dir=./MusicTransformer-tensorflow2.0/music` (ponder-lab/Python-Subjects#196). They are
# not in this repository (`*.pickle` is ignored) and are not re-derived here: re-downloading MIDI
# and re-running preprocess.py would produce a different corpus from the one measured. The corpus's
# fetch script downloads the archived original from Zenodo, verifies it against the checksum
# committed in Python-Subjects, and unpacks it into music/; it is a no-op when that is already done.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
fetch="$here/../scripts/fetch_irreplaceable_assets.sh"
if [[ ! -f "$fetch" ]]; then
    echo "setup.sh: $fetch not found. Run this subject from the Python-Subjects corpus (#196)." >&2
    exit 1
fi
bash "$fetch" musictransformer
