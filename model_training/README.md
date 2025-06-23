# Usage
```
deepspeed train_seq2seq.py --model_name_or_path <model_name_or_path> \
--train_file <path_to_train_file> \
--validation_split_percentage <validation_split_percentage> \
--wandb_project <wandb_project_name> \
--output_dir <path_to_output_dir> \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--warmup_steps 100 \
--num_train_epochs <number_of_epochs> \
--learning_rate 3e-4 \
--logging_dir <path_to_logging_dir> \
--deepspeed <path_to_deepspeed_config>
```