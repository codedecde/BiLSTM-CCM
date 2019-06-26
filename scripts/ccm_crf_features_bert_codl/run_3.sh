PYTHON=/zfsauton3/home/bpatra/miniconda3/bin/python3.6
${PYTHON} -m ccm_model.codl_main --config_file training_configs/ccm_crf_features_bert_codl/ner_type_attr.jsonnet --base_dir ./trained_model_outputs/ccm_crf_features_bert_codl --devices 0 --start_index 105 --end_index 136
