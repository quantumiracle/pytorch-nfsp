echo "Running DATE:" $(date +"%Y-%m-%d %H:%M")

DATE=`date '+%Y%m%d_%H%M'`

#nohup python main.py --env 'SlimeVolleyPixel-v0' --train-freq 1000 --batch-size 256 > log/$DATE$RAND.log &
# nohup python main.py --env 'pong_v1'  > log/$DATE$RAND.log &
# nohup python main.py --env 'pong_v1' --train-freq 1000 --batch-size 256 --eta 1  > log/$DATE$RAND.log &
#python train_dqn_against_baseline.py --env SlimeVolley-v0 > log/$DATE$RAND.log &
python train_dqn_against_baseline.py --env SlimeVolleyPixel-v0 > log/$DATE$RAND.log &
