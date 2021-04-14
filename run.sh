tmux a -t rgnn
cd /home/PublicDir/liuledian/venv/bin
source activate
cd /home/liuledian/github/confidence/
python train.py --proc=0 --z_dim=30

tmux a -t rgnn1
cd /home/PublicDir/liuledian/venv/bin
source activate
cd /home/liuledian/github/confidence/
python train.py --proc=1 --z_dim=30