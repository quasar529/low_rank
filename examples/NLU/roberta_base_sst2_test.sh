python examples/text-classification/run_glue.py \
--model_name_or_path  roberta-base \
--output_dir ./output \
--task_name sst2 \
--do_eval \
--apply_lora \
--lora_r 8 \
--lora_alpha 8 \
--seed 0 \
--ex_type initialize_dW_A_with_svd
#/home/lab/bumjun/lora-training/LoRA/examples/NLU/sst2/lora/2023-09-11_16:37:03lorackpt.bin
#--lora_path /home/lab/bumjun/lora-training/LoRA/examples/NLU/sst2/r16r_no_merged/model/checkpoint-15810/pytorch_model.bin \
#--lora_path /home/lab/bumjun/low_rank/examples/NLU/sst2/lora/2023-09-17_17:47_lorackpt.bin \
#--lora_path /home/lab/bumjun/low_rank/examples/NLU/sst2/r16/model/checkpoint-15810/2023-09-20_15:55_lorackpt.bin \
#--lora_path /home/lab/bumjun/low_rank/examples/NLU/sst2/r16/model/checkpoint-15810/2023-09-20_15:55_lorackpt.bin \
#textattack/roberta-base-SST-2
#--lora_path /home/lab/bumjun/low_rank/examples/NLU/sst2/initialize_dW_with_svd_r1_train_CL/model/checkpoint-4216/2023-10-06_01:23_lorackpt.bin \