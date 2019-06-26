PYTHON=/zfsauton3/home/bpatra/miniconda3/bin/python3.6
${PYTHON} -m ccm_model.main --config_file training_configs/ccm_crf_features_bert_codl/ner_type_attr.jsonnet --base_dir ./trained_model_outputs/ccm_crf_features_bert_codl --devices 0 --start_index 35 --end_index 70
