echo 'start sh'
python run_train.py -rc gru -b 32 -n 512 -f 81 -c 29 -rl 7 -cl 3 -o adam -a relu -lr 1e-3 -gc 5 -p 20 -i 0 -t /home/yb/mywork/asr/libri_featlabel/train-clean-100.json /home/yb/mywork/asr/libri_featlabel/train-clean-360.json -d /home/yb/mywork/asr/libri_featlabel/dev-clean.json -s /home/yb/mywork/asr/savemodels/ds2_0311
