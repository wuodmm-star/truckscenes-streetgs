#!/bin/bash
# Copy the TruckScenes adapter into a drivestudio checkout and register the dataset.
# usage: bash tools/install_addon.sh /path/to/drivestudio
set -eo pipefail
DS=${1:?usage: bash tools/install_addon.sh /path/to/drivestudio}
HERE=$(cd "$(dirname "$0")/.." && pwd)
cp -r "$HERE/addon/datasets/truckscenes" "$DS/datasets/"
mkdir -p "$DS/configs/datasets/truckscenes"
cp "$HERE"/addon/configs/datasets/truckscenes/*.yaml "$DS/configs/datasets/truckscenes/"
cd "$DS" && python "$HERE/tools/patch_drivestudio.py"
echo "TruckScenes adapter installed into $DS"
