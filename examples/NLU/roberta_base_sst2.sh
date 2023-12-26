# export num_gpus=8
# export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
# export PYTHONHASHSEED=0
export output_dir="./sst2/roberta-base_fix_W_initialize_dW_with_svd_r_to_2r"
python /home/lab/bumjun/low_rank/examples/NLU/examples/text-classification/run_glue.py \
--model_name_or_path roberta-base \
--task_name sst2 \
--do_train \
--do_eval \
--max_seq_length 128 \
--per_device_train_batch_size 32 \
--learning_rate 4e-4 \
--num_train_epochs 60 \
--output_dir $output_dir/model \
--logging_dir $output_dir/log \
--overwrite_output_dir \
--logging_steps 10 \
--evaluation_strategy epoch \
--save_strategy epoch \
--apply_lora \
--lora_r 8 \
--lora_alpha 8 \
--seed 0 \
--weight_decay 0.1 \
--ex_type initialize_dW_with_svd_r_to_2r
#--warmup_ratio 0.06 \
#--lora_path /home/lab/bumjun/low_rank/examples/NLU/sst2/r16/model/checkpoint-480/pytorch_model.bin \
# --output_dir $output_dir/model \./sst2/eval_test/model
# --logging_dir $output_dir/log \ ./sst2/eval_test/log
#--lora_path /home/lab/bumjun/low_rank/examples/NLU/sst2/r16/model/checkpoint-15810/2023-09-20_15:55_lorackpt.bin \
#--lora_path /home/lab/bumjun/low_rank/examples/NLU/sst2/r16/model/checkpoint-15810/2023-09-20_15:55_lorackpt.bin \
#--lora_path /home/lab/bumjun/low_rank/examples/NLU/sst2/lora_ft_rank1_roberta-base/model/checkpoint-4216/2023-10-04_21:41_lorackpt.bin \