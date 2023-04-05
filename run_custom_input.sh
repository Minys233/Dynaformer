#! /bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
bash Dynaformer/examples/evaluate/evaluate.sh "" "custom:path=$SCRIPT_DIR/example_data/example.pkl" "" _custom
