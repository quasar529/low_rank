export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir="./output/eval"
python examples/text-classification/run_glue_og.py \
--model_name_or_path microsoft/deberta-v2-xxlarge \
--task_name mrpc \
--do_eval \
--fp16 \
--max_seq_length 128 \
--per_device_eval_batch_size 24 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir/log \
--evaluation_strategy steps \
--eval_steps 10 \
--apply_lora \
--lora_r 4 \
--lora_alpha 16 \
--lora_path home/lab/bumjun/low_rank/examples/NLU/output/init_dW_B_with_svd_by_head-rank4-alpha16-seed42/model/checkpoint-1100 \
--seed 42