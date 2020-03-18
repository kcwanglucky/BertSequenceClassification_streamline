# bert_run_lm_streamline
Streamlining the usage of huggingface run_language_model.py in finetuning or training from scratch a bert model with your own text data

### How to run fine tuning:
python bert_run_lm_streamline --data_path=[data_path] --epoch=30 --batch_size=64 --min_each_group=3 --maxlength=30 --model_output=[model_path] --model_start --model_prediction

### How to do prediction:
python bert_run_lm_streamline --data_path=[data_path_testing] --epoch=30 --batch_size=64 --min_each_group=3 --maxlength=30 --model_start=[model_path] --model_prediction=[prediction_path]
