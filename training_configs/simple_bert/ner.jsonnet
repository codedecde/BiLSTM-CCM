// Configuration for a named entity recognization model based on:
// BERT
local bert_embedding_dim = 768;
{
  "dataset_reader": {
    "type": "feature_reader",
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
  "base_output_dir": "./trained_model_outputs/basic",
  "all_data_path": "./data/Data_136_with_feats.txt",
  "model": {
    "type": "simple_tagger",
    "text_field_embedder": {
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
        "type": "pass_through",
        "input_dim": bert_embedding_dim
    },
    "calculate_span_f1": true,
    "label_encoding": "IOB1"
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
