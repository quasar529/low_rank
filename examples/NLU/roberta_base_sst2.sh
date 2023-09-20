# export num_gpus=8
# export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
# export PYTHONHASHSEED=0
export output_dir="./sst2/r16"
python /home/lab/bumjun/low_rank/examples/NLU/examples/text-classification/run_glue.py \
--model_name_or_path textattack/roberta-base-SST-2 \
--task_name sst2 \
--do_train \
--do_eval \
--max_seq_length 128 \
--per_device_train_batch_size 64 \
--learning_rate 3e-5 \
--num_train_epochs 30 \
--output_dir $output_dir/model \
--logging_dir $output_dir/log \
--overwrite_output_dir \
--logging_steps 10 \
--evaluation_strategy epoch \
--save_strategy epoch \
--apply_lora \
--lora_r 16 \
--lora_alpha 16 \
--seed 0 \
--weight_decay 0.1 \
--ex_type dW=WSVD
#--warmup_ratio 0.06 \
#--lora_path /home/lab/bumjun/low_rank/examples/NLU/sst2/r16/model/checkpoint-480/pytorch_model.bin \
# --output_dir $output_dir/model \./sst2/eval_test/model
# --logging_dir $output_dir/log \ ./sst2/eval_test/log