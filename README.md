# Textless Dependency Parsing by Labeled Sequence Prediction

- Paper URL: TBD

## Setup Environment

- Install `Python 3.9.7`
- Install required packages `pip install -r requirements.txt`
    - Consider using any package manager such as `venv` or `poetry`.
- Install [`sclite`](https://github.com/usnistgov/SCTK)
  - Don't forget to update `PATH`: `export PATH = "/home/Sclite/Path/...:$PATH"`
- Install `sox` command line tool following [this instruction](https://pysox.readthedocs.io/en/latest/)

## Dataset Preparation

Following the intruction, you will obtain the following three types of files:

- Dataset csv file (train, dev, test): Main dataset file used in the training pipeline.
- conllu file (dev, test): Original conllu file.
- conllu_trans file (dev, test): Sentence-only (=transcription) file.

### French (Orféo Treebank)

- Please refer to [wav2tree repository](https://gricad-gitlab.univ-grenoble-alpes.fr/pupiera/wav2tree_release).
- You need an additional preprocessing for generating gold conllu file.  
  Please execute `./data_processing/fr_orfeo/create_gold_conllu_file.py`.

### English (Switchboard)

- Please refer to [data_processing/en_swbd/README.md](./data_processing/en_swbd/README.md).

## Model Training

Most parameters are specified in `hparams/hparams.yaml`.  
You have to specify following extra parameters on command-line.

- `--data_folder`: Path to data_folder, which should include following files:
  - dataset csv file x3 (train, dev, test)
  - conllu file x2 (dev, test)
  - conllu_trans file x2 (dev, test)

Default hparams are set based on English Switchboard corpus.
If you train on the other dataset (including Orféo Treebank), you have to specify following parameters.

- `--dataset`: Dataset name. You can specify it arbitrarily.
- `--wav2vec2_hub`: Pretrained model name in Huggingface. Behavior of specifying the model other than wav2vec2 is not guaranteed.

For example, if you hope to train the model on French Orféo Treebank in the same setting as the paper, run the following.

```
python train.py hparams/hparams.yaml --data_folder /path/to/dataset --dataset fr-orfeo --wav2vec2_hub LeBenchmark/wav2vec2-FR-7K-large
```

## Analysis

The `./analysis` directory contains simple scripts for analysis performed in the paper.

- `pos_analysis.ipynb` corresponds to Section 5.1 "Analysis 1: Prediction Accuracy of Head Position"
- `uas_analysis.py` corresponds to Section 5.2 "Analysis 2: Advantage of Textless Method"

If you hope to try it, please upload variables specified in the script.

## Note: Training and analysis of wav2tree

For training wav2tree with subword vocabulary and generate directories required for analysis,
you will need modification on the original code.  
If you need to make it work, please contact the author

## Author information

```
Shunsuke Kando
https://gifdog97.github.io/
```
