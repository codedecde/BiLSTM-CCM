// Configuration for a named entity recognization model based on:
//   Peters, Matthew E. et al. “Deep contextualized word representations.” NAACL-HLT (2018).
local attr_penalty = 7.2159;
local num_features = 282;
local hidden_size = 10;
{
  "dataset_reader": {
    "type": "handcrafted_feature_reader",
    "features_index_map": "./data/features.txt",
    "coding_scheme": "IOB1",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      },
      "token_characters": {
        "type": "characters",
        "min_padding_length": 3
      }
    }
  },
  "base_output_dir": "./trained_model_outputs/basic",
  "all_data_path": "./data/Data_136_with_feats.txt",
  "partial_data_path": "./data/partially_labeled_data_test.txt",
  "model": {
    "type": "ccm_model",
    "label_encoding": "IOB1",
    "num_features": num_features,
    "constrain_crf_decoding": true,
    "calculate_span_f1": true,
    "dropout": 0.5,
    "include_start_end_transitions": false,
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "embedding",
            "embedding_dim": 50
        },
        "token_characters": {
            "type": "character_encoding",
            "embedding": {
                "embedding_dim": 16
            },
            "encoder": {
                "type": "cnn",
                "embedding_dim": 16,
                "num_filters": 128,
                "ngram_filter_sizes": [3],
                "conv_layer_activation": "relu"
            }
          }
       },
    },
    "encoder": {
        "type": "lstm",
        "input_size": 50 + 128 + num_features,
        "hidden_size": hidden_size,
        "num_layers": 2,
        "dropout": 0.5,
        "bidirectional": true
    },
    "ccm_decoder": {
        "label_namespace": "labels",
        "label_encoding": 'IOB1',
        "constrain_crf_decoding": true,
        "hard_constraints": ["type"],
        "soft_constraints": {"attr": attr_penalty}
    }
  },
  "iterator": {
    "type": "basic",
    "batch_size": 64
  },
  "trainer": {
    "optimizer": {
        "type": "adam",
        "lr": 0.001
    },
    "validation_metric": "-loss",
    "num_epochs": 1,
    "grad_norm": 5.0,
    "cuda_device": -1
  }
}
