#!/bin/bash

DIR="$(cd "$(dirname "$0")" && pwd)"
CUDA_DEVICE_AVAILABLE=false
CUDA_DEVICES=0
REQUIREMENTS_FLAG=false
PYTHON3=false
PYTHON=false
PYTHON_VERSION="null"
DOCKER=false

usage() {
    echo "Usage: $0 [-c <number of devices>] [-r] [-h]"
    exit 1
}

help() {
    echo "= Help ==========================="
    echo "This script checks for various requirements and runs a Python script."
    echo ""
    echo "Options:"
    echo "  -c <number of devices>  Set the number of CUDA devices (default: 0)"
    echo "  -r                     Check requirements (e.g., Docker, Python)"
    echo "  -h                     Display this help message"
    echo ""
    echo "Example usage:"
    echo "  $0 -c 2 -r"
    echo ""
    exit 0
}

# Parse command-line options
while getopts "c:rh" opt; do
    case "$opt" in
        c) CUDA_DEVICES="$OPTARG" ;;
        r) REQUIREMENTS_FLAG=true ;;
        h) help ;;
        ?) usage ;;
    esac
done

# |====================================================|
# | Requirements ****************************************|
# |====================================================|
if [ "$REQUIREMENTS_FLAG" = true ]; then
    echo -e "\nRequirements"
    echo "---------------------------------"

    # Check if Docker exists
    if docker --version >/dev/null 2>&1 && docker-compose --version >/dev/null 2>&1; then
        echo "- Docker installed...  ✅"
    else
        echo "- Docker Not Installed... ❌"
    fi

    # Check if Python (python) is installed and get version
    if command -v python &>/dev/null; then
        echo "- python installed...  ✅"
    fi

    # Check if Python 3 (python3) is installed and get version
    if command -v python3 &>/dev/null; then
        echo "- python3 installed... ✅"
    else
        echo "- python3 is not installed... ❌"
        exit 0
    fi
    echo "---------------------------------"
    exit 0
else
    if command -v python &>/dev/null; then
        PYTHON=true
        PYTHON_VERSION=$(python -V 2>&1 | cut -d " " -f2)
    fi

    # Check if Python 3 (python3) is installed and get version
    if command -v python3 &>/dev/null; then
        PYTHON3=true
        PYTHON_VERSION=$(python3 -V 2>&1 | cut -d " " -f2)
    else
        echo "=== End Program ==="
        exit 0
    fi
fi

# |========================================================================|
# | ******************* Begin Of Program **********************************|
# |========================================================================|
echo "==========================================================================
 ________       ___    ___ ________   _________  ________     ___    ___ 
|\   ____\     |\  \  /  /|\   ___  \|\___   ___\\   __  \   |\  \  /  /|
\ \  \___|_    \ \  \/  / | \  \\ \  \|___ \  \_\ \  \|\  \  \ \  \/  / /
 \ \_____  \    \ \    / / \ \  \\ \  \   \ \  \ \ \   __  \  \ \    / / 
  \|____|\  \    \/  /  /   \ \  \\ \  \   \ \  \ \ \  \ \  \  /     \/  
    ____\_\  \ __/  / /      \ \__\\ \__\   \ \__\ \ \__\ \__\/  /\   \  
   |\_________\\___/ /        \|__| \|__|    \|__|  \|__|\|__/__/ /\ __\ 
   \|_________\|___|/                                        |__|/ \|__| 
=========================================================================="


if [ $PYTHON = true ]; then
    python "$DIR/cli.py"
    exit 0
fi

if [ $PYTHON3 = true ]; then
    python3 "$DIR/cli.py"
else
    exit 0
fi

echo "=== End Program ==="