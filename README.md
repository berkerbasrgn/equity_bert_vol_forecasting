# Vola-BERT

This code repository contains an extend version for our proposed model, Vola-BERT, where we allow for a flexible number of semantic tokens, not limited to market sessions and economic events.
<p align="center">
  <img src="https://gitlab.doc.ic.ac.uk/qn24/vola-bert/-/raw/master/figs/vola-bert-architecture.png?ref_type=heads"/>
</p>

## Model Specification
Semantic tokens are specified via a dictionary mapping each token name to its number of token types. For example, in our initial experiments, we use 5 market sessions and 3 impact events, defining the model as follows:
```
model = Vola_BERT(
          **{
              'input_len': input_len,
              'pred_len': pred_len,
              'n_layer': n_layer,
              'num_series': num_series,
              'semantic_tokens': {'market_session': 5, 'event': 3}
          }
        )
```
For input shapes and training procedure, we provide a demo notebooks containing input examples as well as training code.

## Repository Structure
```
data_preprocessing/
(Pre-processes data from *FirstRate* and merges economic events from Economic Calendar)
├── firstrate_data_processing.ipynb
└── calendar_merging.ipynb     

src/
├── dataset.py                  # Defines PyTorch Dataset for exchange rate data retrieved from *FirstRate*
├── loss.py                     # Loss function considered for training and evaluation
├── utils.py                    # Contains Early Stopping and Standard Scaler utilities for training
├── plot_utils.py               # Contains plot utilities for semantic tokens evaluation
├── trainer.py                  # Implements model training, validation, and early stopping
└── model_bert.py               # Contains implementations of Vola-BERT framework

examples/
├── fomc_press.pkl              # Contains 6 example timepoints corresponding to FOMC Press Conference
└── london_open.pkl             # Contains 6 example timepoints corresponding to London Open (8:00 AM local time)

checkpoints/
└── vola_bert_example/
    └── checkpoint.pth          # an example trained Vola-BERT

demo.ipynb                      # Demo notebook illustrating model specification, training, and
                                  evaluation on example timepoints corresponding to London Open and FOMC Press Conference.
```