import re
import os
scripts_dir = "./books-scripts/grid_search"
config_dir = "./books_training_configs/grid_search"
base_dir = "./books_trained_outputs/grid_search"
os.makedirs(scripts_dir, exist_ok=True)
os.makedirs(os.path.join(scripts_dir, "nohup-outs"), exist_ok=True)
os.makedirs(config_dir, exist_ok=True)
os.makedirs(base_dir, exist_ok=True)

PYTHON = "/zfsauton3/home/bpatra/miniconda3/bin/python3.6"
num_scripts = 6
scripts_to_configs = [[] for _ in range(num_scripts)]
script_number = 0


for attr_penalty in [0, 1.0, 5.0, 10.0]:
    for sent_penalty in [0, 1.0, 5.0, 10.0]:
        buf = f"""
// Configuration for a named entity recognization model based on:
// BERT
local bert_embedding_dim = 768;
local attr_penalty = {attr_penalty};
local num_features = 357;
local sent_penalty = {sent_penalty};
{{
  "dataset_reader": {{
    "type": "handcrafted_feature_reader",
    "coding_scheme": "IOB1",
    "features_index_map": "./books-data/features.txt",
    "token_indexers": {{
      "bert": {{
          "type": "bert-pretrained",
          "pretrained_model": "./data/embeddings/bert/bert-base-multilingual-cased/vocab.txt",
          "do_lowercase": false,
          "use_starting_offsets": true
      }}
    }},
    "use_sentence_markers": true
  }},
  "base_output_dir": "./trained_model_outputs/basic",
  "all_data_path": "./books-data/book.feb.2018.actor_replaced.full.gold",
  "vocab_dir": "./books-data/vocabulary",
  "model": {{
    "type": "ccm_model",
    "label_encoding": "IOB1",
    "constrain_crf_decoding": true,
    "calculate_span_f1": true,
    "num_features": num_features,
    "dropout": 0.5,
    "include_start_end_transitions": true,
    "text_field_embedder": {{
      "type": "basic",
      "token_embedders": {{
        "bert": {{
            "type": "bert-pretrained",
            "pretrained_model": "./data/embeddings/bert/bert-base-multilingual-cased/bert-base-multilingual-cased.tar.gz"
        }},
      }},
      "allow_unmatched_keys": true,
      "embedder_to_indexer_map": {{
            "bert": ["bert", "bert-offsets"]
      }}
    }},
    "encoder": {{
        "type": "lstm",
        "input_size": bert_embedding_dim + num_features,
        "hidden_size": 200,
        "num_layers": 2,
        "dropout": 0.5,
        "bidirectional": true
    }},
    "ccm_decoder": {{
        "label_namespace": "labels",
        "label_encoding": 'IOB1',
        "constrain_crf_decoding": true,
        "hard_constraints": ["type"],
        "soft_constraints": {{"attr": attr_penalty}},
        "sentence_penalty_map": {{"I-type": sent_penalty}}
    }}
  }},
  "iterator": {{
    "type": "basic",
    "batch_size": 8
  }},
  "trainer": {{
    "optimizer": {{
        "type": "adam",
        "lr": 0.001
    }},
    "validation_metric": "-loss",
    "num_epochs": 25,
    "grad_norm": 5.0,
    "patience": 10,
    "cuda_device": 0
  }}
}}
"""
        filename = re.sub(r"\.", "-", f"config_attr_penalty_{attr_penalty}_sent_penalty_{sent_penalty}") + ".jsonnet"
        filename = os.path.join(config_dir, filename)
        with open(filename, "w") as outfile:
            outfile.write(buf)
        scripts_to_configs[script_number].append(filename)
        script_number = (script_number + 1) % (num_scripts)

for index in range(len(scripts_to_configs)):
    outbuf = []
    for file in scripts_to_configs[index]:
        _, name = os.path.split(file)
        name = re.sub(r"\.jsonnet", "", name)
        model_dir = os.path.join(base_dir, name)
        outbuf.append(f"{PYTHON} -m ccm_model.main --config_file {file} --base_dir {model_dir} --devices 0 --start_index 0 --end_index 92")
    with open(os.path.join(scripts_dir, f"run_{index}.sh"), "w") as outfile:
        outfile.write("\n".join(outbuf))
