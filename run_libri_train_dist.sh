echo 'start distributed training'
echo $1 $2
python run_train.py --ps_hosts localhost:2222 --ws_hosts localhost:2223,localhost:2224 --job_name $1 --task_index $2 -us no -rc gru -b 64 -n 512 -f 81 -c 29 -rl 5 -cl 3 -o adam -a relu -lr 5e-5 -gc 5 -p 30 -i 10 -t /home/yb/mywork/asr/libri_featlabel/train-clean-100.json /home/yb/mywork/asr/libri_featlabel/train-clean-360.json -d /home/yb/mywork/asr/libri_featlabel/dev-clean.json -s /home/yb/mywork/asr/savemodels/ds2_1201_mgpu -gf 1.0
echo 'end distributed training'