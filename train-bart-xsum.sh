nohup python trainer.py \
--max_seq_len 710 \
--sep-token [SEP] \
--data-dir summarisation_data/ \
--num_train_epochs 5 \
--eval_steps 1000 \
--lr_scheduler_type cosine \
--learning_rate 5e-5 \
--warmup_ratio 0.30 \
--per_device_train_batch_size 10 \
--per_device_eval_batch_size 10 \
--save_total_limit 1 \
--model_base  facebook/bart-base \
--run_id bart_base_model_full \
--is-not-auto-encoder-data \
--output_dir trained_models_sum/  >> training_logs_bart_xsum.out &