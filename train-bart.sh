nohup python trainer.py \
--max_seq_len 700 \
--num_train_epochs 5 \
--eval_steps 1000 \
--lr_scheduler_type cosine \
--learning_rate 5e-5 \
--warmup_ratio 0.15 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--fp16 \
--gradient-accumulation-steps 4 \
--save_total_limit 1 \
--model_base  facebook/bart-base \
--run_id bart_base_model_full \
--output_dir trained_models_mtl/  >> training_logs_bart_full.out &