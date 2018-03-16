echo 'start listener attention speller cluster training ...'
echo $1 $2
[ $1 = 'ps' ] && export CUDA_VISIBLE_DEVICES= && python run_train.py --ps_hosts localhost:2232 --ws_hosts localhost:2233,localhost:2234 --job_name $1 --task_index $2 -rc gru -b 32 -n 512 -f 81 -c 29 -rl 7 -cl 3 -o adam -a relu -lr 1e-3 -gc 5 -p 20 -i 0 -md las -t /home/yb/mywork/asr/libri_featlabel/train-clean-100.json /home/yb/mywork/asr/libri_featlabel/train-clean-360.json -d /home/yb/mywork/asr/libri_featlabel/dev-clean.json -s /home/yb/mywork/asr/savemodels/las_dist_1214
[ $1 = 'worker' ] && export CUDA_VISIBLE_DEVICES=$2 && python run_train.py --ps_hosts localhost:2232 --ws_hosts localhost:2233,localhost:2234 --job_name $1 --task_index $2 -rc gru -b 32 -n 512 -f 81 -c 29 -rl 7 -cl 3 -o adam -a relu -lr 1e-3 -gc 5 -p 20 -i 6 -md las -t /home/yb/mywork/asr/libri_featlabel/train-clean-100.json /home/yb/mywork/asr/libri_featlabel/train-clean-360.json -d /home/yb/mywork/asr/libri_featlabel/dev-clean.json -s /home/yb/mywork/asr/savemodels/las_dist_1214



echo 'end listener attention speller cluster training ...'
