#!/usr/bin/env bash
set -euo pipefail

# Protect AdminConfig.json so only root can modify it (readable by group).
# Usage: sudo ./scripts/protect_admin_config.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
FILE_PATH="${ROOT_DIR}/AdminConfig.json"

if [[ ! -f "${FILE_PATH}" ]]; then
  echo "ERROR: ${FILE_PATH} not found."
  exit 1
fi

# Change ownership to root and restrict permissions to 640 (rw-r-----)
# Requires sudo privileges.
sudo chown root:root "${FILE_PATH}"
sudo chmod 640 "${FILE_PATH}"

echo "Protected ${FILE_PATH}:"
ls -l "${FILE_PATH}"
