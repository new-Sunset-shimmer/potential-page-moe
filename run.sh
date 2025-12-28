CUDA_VISIBLE_DEVICES=4 nohup python main_7b.py >> eval_8x22B_new.log
wait
CUDA_VISIBLE_DEVICES=4 nohup python main_22b.py >> eval_8x7B_new.log