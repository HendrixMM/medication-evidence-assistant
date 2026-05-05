#!/usr/bin/env bash
set -euo pipefail

echo "Checking environment setup..."

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
  exit 1
fi

if [ -n "${PUBMED_EMAIL:-}" ]; then
  echo "PUBMED_EMAIL is set."
else
  echo "PUBMED_EMAIL is not set. PubMed requests work better with a contact email."
fi

echo "Environment looks ready."
