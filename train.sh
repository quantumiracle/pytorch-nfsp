echo "Running DATE:" $(date +"%Y-%m-%d %H:%M") 
DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE

nohup python main.py --env 'SlimeVolley-v0' --hidden-dim 256 --max-frames 20000000 > log/$DATE$RAND.log &
# nohup python main.py --env 'SlimeVolleyNoFrameskip-v0' --hidden-dim 512 --max-frames 30000000 > log/$DATE$RAND.log &
#nohup python main.py --env 'SlimeVolleyNoFrameskip-v0' --train-freq 1000 --batch-size 256 > log/$DATE$RAND.log &
#nohup python main.py --env 'pong_v1' --max-frames 20000000 --ram  > log/$DATE$RAND.log &
# nohup python main.py --env 'pong_v1' --train-freq 1000 --batch-size 256 --eta 1  > log/$DATE$RAND.log &
# python train_dqn_against_baseline.py --env SlimeVolley-v0 --hidden-dim 256 --max-frames 10000000 > log/$DATE$RAND.log &
#python train_dqn_against_baseline.py --env SlimeVolleyNoFrameskip-v0 --hidden-dim 512 --max-frames 30000000 > log/$DATE$RAND.log &
