# Multilingual speech processing
### Subtask 1:
Build an ASR model for the Guarani Dataset using Fairseq

#### Run Instructions
1. Load the Assignment_2_fairseq.ipynb into colab
2. Download the Guarani dataset here -> https://commonvoice.mozilla.org/en/datasets
3. Create a folder named 'commonvoice' and unzip the dataset into the newly created folder
4. Clone the fairseq repository into colab, navigate to /examples/speech_to_text/
5. Add the files ```prep_librispeech_data.py``` and ```librispeech_art.py``` into the folder
6. Run the cells sequentially to preprocess, train and evaluate fairseq models.
7. Please note, in order to connect to wandb, you must replace the API-key while calling ```wandb.init()```

### Subtask 2:
Finetuning the pre-trained model XLSR/Wave2Vec2 for ASR for Guarani

#### Run Instructions

