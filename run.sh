tmux a -t rgnn
cd /home/PublicDir/liuledian/venv/bin
source activate
cd /home/liuledian/github/confidence/
python train.py --proc=0 --task=animal --subject=wuxin

tmux a -t rgnn1
cd /home/PublicDir/liuledian/venv/bin
source activate
cd /home/liuledian/github/confidence/
python train.py --proc=1 --task=animal --subject=wuxin