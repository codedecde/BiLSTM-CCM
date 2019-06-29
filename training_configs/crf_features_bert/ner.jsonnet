// Configuration for a named entity recognization model based on:
// BERT
local bert_embedding_dim = 768;
local num_features = 282;
{
  "dataset_reader": {
    "type": "handcrafted_feature_reader",
    "features_index_map": "./data/features.txt",
    "coding_scheme": "IOB1",
    "token_indexers": {
      "bert": {
          "type": "bert-pretrained",
          "pretrained_model": "./data/embeddings/bert/bert-base-multilingual-cased/vocab.txt",
          "do_lowercase": false,
          "use_starting_offsets": true
      }
    }
  },
  "base_output_dir": "./trained_model_outputs/crf_features_bert",
  "all_data_path": "./data/Data_136_with_feats.txt",
  "model": {
    "type": "ccm_model",
    "label_encoding": "IOB1",
    "constrain_crf_decoding": true,
    "calculate_span_f1": true,
    "num_features": 282,
    "dropout": 0.5,
    "include_start_end_transitions": false,
    "text_field_embedder": {
      "type": "basic",
      "token_embedders": {
        "bert": {
            "type": "bert-pretrained",
            "pretrained_model": "./data/embeddings/bert/bert-base-multilingual-cased/bert-base-multilingual-cased.tar.gz"
        },
      },
      "allow_unmatched_keys": true,
      "embedder_to_indexer_map": {
            "bert": ["bert", "bert-offsets"]
      }
    },
    "encoder": {
        "type": "lstm",
        "input_size": bert_embedding_dim + num_features,
        "hidden_size": 200,
        "num_layers": 2,
        "dropout": 0.5,
        "bidirectional": true
    },
  },
  "iterator": {
    "type": "basic",
    "batch_size": 8
  },
  "trainer": {
    "optimizer": {
        "type": "adam",
        "lr": 0.001
    },
    "validation_metric": "-loss",
    "num_epochs": 25,
    "grad_norm": 5.0,
    "patience": 10,
    "cuda_device": 0
  }
}
