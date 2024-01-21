export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir="./output/STS_B/deberta_init_dW_B_T_with_svd_us_by_head_scaling-rank24-alpha24-seed0"
python examples/text-classification/run_glue_og.py \
--model_name_or_path microsoft/deberta-v2-xxlarge \
--task_name stsb \
--do_train \
--do_eval \
--max_seq_length 128 \
--per_device_train_batch_size 16 \
--learning_rate 2e-4 \
--num_train_epochs 10 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir/log \
--fp16 \
--evaluation_strategy steps \
--eval_steps 50  \
--metric_for_best_model pearson \
--save_strategy steps \
--save_steps 50 \
--save_total_limit 2 \
--warmup_steps 100 \
--apply_lora \
--lora_r 24 \
--lora_alpha 24 \
--seed 0 \
--weight_decay 0.1 \
--ex_type deberta_init_dW_B_T_with_svd_us_by_head_scaling