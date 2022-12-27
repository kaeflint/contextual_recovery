nohup python trainer.py \
--max_seq_len 512 \
--num_train_epochs 5 \
--eval_steps 1000 \
--lr_scheduler_type cosine \
--learning_rate 3e-5 \
--warmup_ratio 0.25 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--save_total_limit 1 \
--model_base  facebook/bart-base \
--run_id bart_base_model_1 \
--output_dir trained_models_mtl/  >> training_logs_bart1.out &