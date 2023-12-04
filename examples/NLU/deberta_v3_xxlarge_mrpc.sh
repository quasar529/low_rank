
export output_dir="./mrpc"
python examples/text-classification/run_glue.py \
--model_name_or_path microsoft/deberta-v3-base \
--task_name mrpc \
--do_train \
--do_eval \
--max_seq_length 128 \
--per_device_train_batch_size 64 \
--learning_rate 2e-4 \
--num_train_epochs 30 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir/log \
--fp16 \
--evaluation_strategy steps \
--eval_steps 10 \
--save_strategy steps \
--save_steps 10 \
--warmup_ratio 0.1 \
--apply_lora \
--lora_r 8 \
--lora_alpha 8 \
--seed 0 \
--weight_decay 0.01 \
--ex_type ft_with_lora
