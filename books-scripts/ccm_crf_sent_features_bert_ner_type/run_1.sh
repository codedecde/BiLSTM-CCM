MODEL=ccm_crf_sent_features_bert
CONFIG_FILE=books_training_configs/${MODEL}/ner_type.jsonnet
OUTPUT_DIR=./books_trained_model_outputs/${MODEL}_ner_type
START=23
END=46
PYTHON=/zfsauton3/home/bpatra/miniconda3/bin/python3.6
${PYTHON} -m ccm_model.main --config_file ${CONFIG_FILE} --base_dir ${OUTPUT_DIR} --devices 0 --start_index ${START} --end_index ${END}