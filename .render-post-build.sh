#!/usr/bin/env bash
echo "Running post-build script to download spaCy model..."
python -m spacy download en_core_web_sm
echo "spaCy model download finished."