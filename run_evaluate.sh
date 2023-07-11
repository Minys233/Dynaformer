#! /bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# for evaluation, we only use coreset (CASF) 2013 version or 2016 version
# thus it is same for choosing refined-set/general-set and seed
# use cutoffs=5-5-5, models are trained on this setting.

save_dir=$(readlink -f './checkpoint')
dataset='pdbbind:set_name=refined-set-2019-coreset-2016,cutoffs=5-5-5,seed=0'
data_path=$(readlink -f './data')
suffix='_CASF2016'

# CASF-2016
bash Dynaformer/examples/evaluate/evaluate.sh $save_dir $dataset $data_path $suffix

# CASF-2013
dataset='pdbbind:set_name=refined-set-2019-coreset-2013,cutoffs=5-5-5,seed=0'
data_path=$(readlink -f './data')
suffix='_CASF2013'
bash Dynaformer/examples/evaluate/evaluate.sh $save_dir $dataset $data_path $suffix

