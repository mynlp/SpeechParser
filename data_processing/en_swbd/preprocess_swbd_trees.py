"""Preprocess swbd trees
- input(s):  mrg file (PTB-formatted tree of swbd instance)
- output(s): preprocessed tree file + corresponding id file

For converting into  preprocess involves following procedures.
- lower-casing texts
- removing punctuation
- removing special EOS tokens (E_S and N_S)
"""

import os
import re
import argparse
from nltk.tree import Tree
from pathlib import Path


def sanitize_tree_str(tree_str: str):
    tree_str = (tree_str
                .replace("(. .)", "")
                .replace("(. ?)", "")
                .replace("(-DFL- E_S)", "")
                .replace("(-DFL- N_S)", ""))
    return re.sub(r'\([^()]+ ,\)', "", tree_str)


def lower_leaves(tree):
    for i in range(len(tree.leaves())):
        position = tree.leaf_treeposition(i)
        tree[position] = tree[position].lower()
    return tree


def preprocess_tree(tree_str: str):
    sanitized = sanitize_tree_str(tree_str)
    tree = Tree.fromstring(sanitized, remove_empty_top_bracketing=True)
    return lower_leaves(tree)


def preprocess_file(mrg_file, out_dir):
    fin = open(mrg_file)
    file_id = mrg_file.stem

    fout_id = open(f"{out_dir}/ids/{file_id}.ids", mode='w')
    fout_mrg = open(f"{out_dir}/mrg/{file_id}.mrg", mode='w')

    idx = 0
    A = "A"
    B = "B"
    speaker = A
    tree_str = ""
    for line in fin:
        # at id declaration line
        if line.startswith("( (CODE"):
            turn_id = Tree.fromstring(line).leaves()[0]
            if turn_id.startswith("SpeakerA"):
                speaker = A
            elif turn_id.startswith("SpeakerB"):
                speaker = B
            else:
                raise ValueError
            continue
        # annotation starts
        if line.startswith("(") and not tree_str:
            tree_str = line
        # annotation continues
        elif line.lstrip().startswith("(") or line.lstrip().startswith(")"):
            tree_str += line
        try:
            tree = preprocess_tree(tree_str)
            sent_id = f"{file_id}_{speaker}_"
            idx += 1
            sent_id += str(idx)
            fout_id.write(f"{sent_id}\n")
            fout_mrg.write(f"{str(tree)}\n")
            tree_str = ""
        # tree annotation is in the middle
        except ValueError:
            continue

    fin.close()
    fout_id.close()
    fout_mrg.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--treebank3_dir',
                        help='directory of treebank3')
    parser.add_argument('--out_dir',
                        help='directory to output preprocessed trees')
    args = parser.parse_args()

    in_dirs = [f'{args.treebank3_dir}/parsed/mrg/swbd/{i}' for i in [2, 3, 4]]
    os.makedirs(f"{args.out_dir}/ids", exist_ok=True)
    os.makedirs(f"{args.out_dir}/mrg", exist_ok=True)
    os.makedirs(f"{args.out_dir}/conllu", exist_ok=True)

    for in_dir in in_dirs:
        for mrg_file in Path(in_dir).iterdir():
            preprocess_file(mrg_file, args.out_dir)
