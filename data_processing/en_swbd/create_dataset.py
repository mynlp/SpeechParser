"""Create Dataset for Training (csv data)
- input(s):  ids file + conllu file
- output(s): Dataset csv file

Csv columns are:
- ID
- duration
- wav (path for wav file)
- wrd
- pos
- gov (0-indexed head id)
- dep
"""
import argparse
from pathlib import Path
import pyconll
import sox
import random
from logging import getLogger, StreamHandler, Formatter, DEBUG

random.seed(1234)

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
handler.setFormatter(Formatter("%(asctime)s %(levelname)s %(name)s :%(message)s"))
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False


class Columns:
    def __init__(self, ID=None, duration=None, wav=None, wrd=None, pos=None, gov=None, dep=None):
        self.ID = ID
        self.duration = duration
        self.wav = wav
        self.wrd = wrd
        self.pos = pos
        self.gov = gov
        self.dep = dep

    def to_csv_line(self):
        return ",".join([
            self.ID,
            str(self.duration),
            self.wav,
            self.wrd,
            self.pos,
            self.gov,
            self.dep
        ]) + "\n"

    def from_annotations(self, sent_id, conllu, wav_file):
        return Columns(
            ID=sent_id,
            duration=self.get_duration(wav_file),
            wav=wav_file,
            wrd=self.get_wrd_seq(conllu),
            pos=self.get_pos_seq(conllu),
            gov=self.get_gov_seq(conllu),
            dep=self.get_dep_seq(conllu)
        )

    def get_duration(self, wav_file):
        return sox.file_info.duration(wav_file)

    def get_wrd_seq(self, conllu: pyconll.unit.sentence.Sentence):
        return ' '.join([token.form for token in conllu])

    def get_pos_seq(self, conllu: pyconll.unit.sentence.Sentence):
        return ' '.join([token.upos for token in conllu])

    def get_gov_seq(self, conllu: pyconll.unit.sentence.Sentence):
        return ' '.join([token.head for token in conllu])

    def get_dep_seq(self, conllu: pyconll.unit.sentence.Sentence):
        return ' '.join([token.deprel for token in conllu])


def generate_trans_conllu(sent_id: str, conllu: pyconll.unit.sentence.Sentence):
    text = " ".join([token.form for token in conllu])
    trans = text + ' (' + sent_id + ')' + '\n'
    conllu_lines = "# sent_id = " + sent_id + "\n" + "# text = " + text + "\n" + conllu.conll() + "\n\n"
    return trans, conllu_lines


def generate_wav_file_path(conversation_id, sent_id, swbd_trim_dir):
    sent_num: str = sent_id.split("_")[-1]
    return f"{swbd_trim_dir}/{conversation_id}/{sent_num.strip().zfill(3)}.wav"


def create_dataset(treebank3_converted_dir, swbd_trim_dir, output_dir):
    ids_dir = f"{treebank3_converted_dir}/ids_fixed"
    conllu_dir = f"{treebank3_converted_dir}/conllu"
    train = open(f"{output_dir}/train.csv", mode="w")
    dev = open(f"{output_dir}/dev.csv", mode="w")
    test = open(f"{output_dir}/test.csv", mode="w")
    dev_conllu = open(f"{output_dir}/dev.conllu", mode="w")
    test_conllu = open(f"{output_dir}/test.conllu", mode="w")
    dev_trans = open(f"{output_dir}/dev.conllu_trans", mode="w")
    test_trans = open(f"{output_dir}/test.conllu_trans", mode="w")

    train.write("ID,duration,wav,wrd,pos,gov,dep\n")
    dev.write("ID,duration,wav,wrd,pos,gov,dep\n")
    test.write("ID,duration,wav,wrd,pos,gov,dep\n")

    for ids_file in Path(ids_dir).iterdir():
        with open(ids_file) as f:
            ids_lines = f.readlines()
        file_id = ids_file.stem
        fconllu = pyconll.load_from_file(f"{conllu_dir}/{file_id}.conllu")
        assert len(ids_lines) == len(fconllu), logger.error(file_id, len(ids_lines), len(fconllu))
        for sent_id, conllu in zip(ids_lines, fconllu):
            wav_file = generate_wav_file_path(file_id, sent_id, swbd_trim_dir)
            try:
                if len(conllu) == 1:
                    logger.info(f"sent_id {sent_id.strip()} skipped (one word sentence)")
                    continue
                columns = Columns().from_annotations(sent_id.strip(), conllu, wav_file)
                csv_line = columns.to_csv_line()
                # train:dev:test = 8:1:1
                val = random.uniform(0, 1)
                if val < 0.8:
                    train.write(csv_line)
                elif val < 0.9:
                    dev.write(csv_line)
                    trans, conllu = generate_trans_conllu(sent_id.strip(), conllu)
                    dev_trans.write(trans)
                    dev_conllu.write(conllu)
                else:
                    test.write(csv_line)
                    trans, conllu = generate_trans_conllu(sent_id.strip(), conllu)
                    test_trans.write(trans)
                    test_conllu.write(conllu)
            except Exception as e:
                logger.info(f"sent_id {sent_id.strip()} skipped due to: {e}")
                # logger.debug(conllu.conll())
                continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--treebank3_converted_dir',
                        help='directory of converted treebank3')
    parser.add_argument('--swbd_trim_dir',
                        help='directory of trimmed switchboard wav')
    parser.add_argument('--output_dir',
                        help='output directory (=csv-formatted dataset used for training and evaluation')
    args = parser.parse_args()
    create_dataset(
        args.treebank3_converted_dir,
        args.swbd_trim_dir,
        args.output_dir
    )
