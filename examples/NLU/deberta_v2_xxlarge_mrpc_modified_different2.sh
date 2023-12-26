
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir="./output/normal-qkv-rank4-alpha4-seed42"
python examples/text-classification/run_glue_og.py \
--model_name_or_path microsoft/deberta-v2-xxlarge \
--task_name mrpc \
--do_train \
--do_eval \
--max_seq_length 128 \
--per_device_train_batch_size 24 \
--learning_rate 2e-4 \
--num_train_epochs 30 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir/log \
--evaluation_strategy steps \
--eval_steps 10 \
--save_strategy steps \
--save_steps 100 \
--apply_lora \
--lora_r 4 \
--lora_alpha 4 \
--seed 42 \
--save_total_limit 2 \
--ex_type normal-qkv-rank4-alpha4-seed42 \
