#!/bin/bash
# Street Gaussians on several TruckScenes scenes, one after another:
#   preprocess -> sky + fine dynamic masks (SegFormer) -> train/eval (2 front cameras, every 8th timestep held out)
# Each step is skipped if its output is already complete, so the script can simply be re-run after an interruption.
#
# usage: bash scripts/run_batch.sh 2 3 4 6 9 0 8
# env (defaults in brackets):
#   DS          drivestudio checkout with the adapter installed   [~/gs/drivestudio]
#   TS_ROOT     TruckScenes root (contains v1.2-mini/, samples/, sweeps/)  [~/gs/data/truckscenes/man-truckscenes]
#   TS_PY       python with truckscenes-devkit                     [python]
#   SEGFORMER   SegFormer checkout                                 [~/gs/SegFormer]
#   OUT_ROOT    training output root                               [~/gs/out/drivestudio]
#   TIMELINE    camera (frame times = real camera capture times) | uniform (v1)   [camera]
#   DATA        processed-data folder, relative to DS              [data/truckscenes/processed_<TIMELINE>/mini]
#   TAG         suffix of the run name                             [empty]
source ${CONDA_SH:-~/miniforge3/etc/profile.d/conda.sh}
[ -d /usr/lib/wsl/lib ] && export LD_LIBRARY_PATH=/usr/lib/wsl/lib   # WSL: CUDA driver libs
DS=${DS:-~/gs/drivestudio}; TS_ROOT=${TS_ROOT:-~/gs/data/truckscenes/man-truckscenes}; TS_PY=${TS_PY:-python}
SEGFORMER=${SEGFORMER:-~/gs/SegFormer}; OUT_ROOT=${OUT_ROOT:-~/gs/out/drivestudio}
export PYTHONPATH=$DS
TIMELINE=${TIMELINE:-camera}; DATA=${DATA:-data/truckscenes/processed_$TIMELINE/mini}; TAG=${TAG:-}
LOG=$OUT_ROOT/truckscenes/logs; mkdir -p $LOG
say() { echo "$(date +%T) $*" | tee -a $LOG/progress.txt; }

for S in "$@"; do
  P=$(printf "%03d" $S); cd $DS
  say "scene $S: preprocess"
  [ -d $DATA/$P/lidar ] || $TS_PY datasets/truckscenes/truckscenes_preprocess.py --root $TS_ROOT --out $DATA --scenes $S --timeline $TIMELINE \
      > $LOG/prep_$P.log 2>&1 || { say "scene $S: PREPROCESS FAILED"; continue; }

  say "scene $S: masks"
  N=$(ls $DATA/$P/images | wc -l)
  if [ $(ls $DATA/$P/fine_dynamic_masks/all 2>/dev/null | wc -l) -lt $N ] || [ $(ls $DATA/$P/sky_masks 2>/dev/null | wc -l) -lt $N ]; then
    conda activate segformer
    python datasets/tools/extract_masks.py --data_root $DATA --segformer_path $SEGFORMER --scene_ids $S \
        --process_dynamic_mask > $LOG/masks_$P.log 2>&1 || { say "scene $S: MASKS FAILED"; conda deactivate; continue; }
    conda deactivate
  fi

  say "scene $S: train"
  conda activate drivestudio
  # render_full / render_novel are off: they keep every rendered frame in RAM (a 392-frame highway scene
  # exhausted 6 GB RAM + 8 GB swap). Test metrics and test-set videos are still produced.
  python tools/train.py --config_file configs/streetgs.yaml --output_root $OUT_ROOT --project truckscenes \
      --run_name streetgs_2cams_s$S$TAG dataset=truckscenes/2cams data.data_root=$DATA data.scene_idx=$S data.pixel_source.test_image_stride=8 \
      render.render_full=False render.render_novel=null \
      > $LOG/train_$P.log 2>&1 && say "scene $S: DONE" || say "scene $S: TRAIN FAILED"
  conda deactivate
done
say "ALL FINISHED"
