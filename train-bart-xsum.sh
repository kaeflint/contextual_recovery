nohup python trainer.py \
--max_seq_len 720 \
--sep-token [SEP] \
--data-dir summarisation_data/ \
--num_train_epochs 5 \
--eval_steps 1000 \
--lr_scheduler_type cosine \
--learning_rate 3e-5 \
--warmup_ratio 0.30 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--save_total_limit 1 \
--model_base  facebook/bart-large \
--run_id bart_large_model_full_e10 \
--is-not-auto-encoder-data \
--output_dir trained_models_sum/  >> training_logs_bart_large_xsum.out &