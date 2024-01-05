# How to create dataset from switchboard corpus

In executing python commands, change pathes depending on your environment.

0. Preparation
   1. Make sure that you've installed all packages in `../../requirements.txt`.
   2. Download following corpora from LDC.
       - [Switchboard-1 Release 2](https://catalog.ldc.upenn.edu/LDC97S62): Switchboard audio files.
       - [Treebank-3](https://catalog.ldc.upenn.edu/LDC99T42): Phrase structure annotations of Switchboard.
       - [NXT Switchboard Annotations](https://catalog.ldc.upenn.edu/LDC2009T26): Detailed annotations of Switchboard.
   3. Download Stanford Parser for dependency tree conversion:  
      https://nlp.stanford.edu/software/lex-parser.shtml
   4. Install sox command line tool on your local, which is used for trimming the wav audio file:  
      https://pysox.readthedocs.io/en/latest/
1. Preprocess `mrg` files in `Treebank-3`.  
Following script generates sanitized mrg files and corresponding id files.
```
python preprocess_swbd_trees.py --treebank3_dir /path/to/treebank_3 --out_dir /path/to/converted_treebank
```
2. Run following script in the stanford-parser directory to obtain dependency trees.

```
data_dir=/path/to/converted_treebank

for mrg in $data_dir/mrg/*
do
  stem="$(basename "$mrg" .mrg)"
	java -mx1g -cp "*" edu.stanford.nlp.trees.ud.UniversalDependenciesConverter -treeFile $mrg > $data_dir/conllu/$stem.conllu
done
```
3. Trim the switchboard wav audio file.
```
python trim_swbd_audio.py \
   --nxt_terminals_dir /path/to/nxt_switchboard_ann/xml/terminals \
   --treebank3_converted_dir /path/to/converted_treebank \
   --swbd_audio_dir /path/to/swb1_LDC97S62 \
   --swbd_trim_dir /path/to/output/trimmed/swbd/audio
```
4. Create final dataset, which will be used for training and evaluation.
```
python create_dataset.py \
   --treebank3_converted_dir /path/to/converted_treebank \
   --swbd_trim_dir /path/to/trimmed/swbd/audio
   --output_dir /path/to/output/final/dataset
```