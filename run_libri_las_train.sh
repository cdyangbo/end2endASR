echo 'start listener attention speller training ...'
export CUDA_VISIBLE_DEVICES=0 && python run_train.py -rc gru -b 32 -n 512 -f 81 -c 29 -rl 7 -cl 3 -o adam -a relu -lr 1e-4 -gc 5 -p 20 -i 0 -md las -t /home/yb/mywork/asr/libri_featlabel/train-clean-100.json /home/yb/mywork/asr/libri_featlabel/train-clean-360.json -d /home/yb/mywork/asr/libri_featlabel/dev-clean.json -s ../savemodels/las_0317
