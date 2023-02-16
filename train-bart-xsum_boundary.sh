nohup python trainer.py \
--max_seq_len 720 \
--sep-token [SEP] \
--data-dir summarisation_data2/ \
--num_train_epochs 10 \
--eval_steps 1000 \
--lr_scheduler_type cosine \
--learning_rate 4e-5 \
--warmup_ratio 0.10 \
--per_device_train_batch_size 12 \
--per_device_eval_batch_size 12 \
--save_total_limit 1 \
--model_base  facebook/bart-base \
--run_id bart_base_model_full_e10a4fp16 \
--is-not-auto-encoder-data \
--fp16 \
--use_random_restrictive \
--gradient-accumulation-steps 4 \
--output_dir trained_models_sum_boundary/  >> training_logs_bart_base_xsum_fp16a_boundary.out &