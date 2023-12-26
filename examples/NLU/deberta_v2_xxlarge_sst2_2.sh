
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir="./output/sst2/normal-rank16-alpha16-seed0"
python examples/text-classification/run_glue_og.py \
--model_name_or_path microsoft/deberta-v2-xxlarge \
--task_name sst2 \
--do_train \
--do_eval \
--max_seq_length 128 \
--per_device_train_batch_size 16 \
--learning_rate 6e-5 \
--num_train_epochs 16 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir/log \
--fp16 \
--evaluation_strategy steps \
--eval_steps 100 \
--save_strategy steps \
--save_steps 100 \
--save_total_limit 2 \
--warmup_steps 1000 \
--apply_lora \
--lora_r 16 \
--lora_alpha 16 \
--seed 0 \
--weight_decay 0.01 \
--ex_type normal-rank16-alpha16-seed0