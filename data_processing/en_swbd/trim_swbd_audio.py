'''
TBD: doc

You need to install sox command line tool on your local: https://pysox.readthedocs.io/en/latest/
'''

import os
import argparse
from pathlib import Path
from collections import defaultdict
import xml.etree.ElementTree as ET
import sox

A = 'A'
B = 'B'
NA = 'n/a'


def opposite(speaker):
    if speaker == A:
        return B
    else:
        return A


def id_changed(child, sentence_id, nite):
    return sentence_id != child.attrib[nite + 'id'].split('_')[0][1:]


def isfloat(s):
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True


def trim_clip(start_end_dict, speaker, swbd_trimmed_dir, audio_file_path):
    for sent_id, start_end in start_end_dict.items():
        start = start_end['start']
        end = start_end['end']
        if (not isfloat(start)) or (not isfloat(end)):
            print(
                f'Time annotation invalid. Not creating file for phrase {sent_id}. start {start}, end : {end}')
            continue
        if float(start) >= float(end):
            print(
                f'Time annotation invalid. Not creating file for phrase {sent_id}. start {start}, end : {end}')
            continue
        tfm = sox.Transformer()
        tfm.trim(start_time=float(start), end_time=float(end))
        tfm.channels(2)
        if speaker == A:
            tfm.remix({1: [1]}, num_output_channels=1)
        else:
            tfm.remix({1: [2]}, num_output_channels=1)
        tfm.set_output_format(rate=16000)
        audio_out_path = f'{swbd_trimmed_dir}/{file_id}/{sent_id.zfill(3)}.wav'
        tfm.build_file(audio_file_path, audio_out_path)


def update_start_end_dict(file_id, id_line, tree_dict, idx_dict, start_end_dict, f_fixed):
    # This block contains ad-hoc processing for taking alignment of NXT annotation and treebank3 annotation

    nite = tree_dict[A].tag.rstrip('terminal_s')

    _, speaker, sent_id = id_line.strip().split('_')
    tree = tree_dict[speaker]
    idx = idx_dict[speaker]

    if idx >= len(tree):
        speaker = opposite(speaker)
        tree = tree_dict[speaker]
        idx = idx_dict[speaker]
    child = tree[idx]

    # id disagrees
    nxt_sent_id = child.attrib[nite + 'id'].split('_')[0]
    if sent_id.strip() != nxt_sent_id[1:]:
        speaker = opposite(speaker)
        tree = tree_dict[speaker]
        idx = idx_dict[speaker]
    child = tree[idx]

    start_end_dict[file_id][speaker][sent_id] = {'start': NA, 'end': NA}
    while not id_changed(child, sent_id, nite) and (idx < len(tree)):
        child = tree[idx]
        while child.tag != 'word':
            idx += 1
            if idx == len(tree):
                break
            child = tree[idx]
            if id_changed(child, sent_id, nite):
                break
        if id_changed(child, sent_id, nite) or idx == len(tree):
            break
        if start_end_dict[file_id][speaker][sent_id]['start'] == NA:
            start_end_dict[file_id][speaker][sent_id]['start'] = child.attrib[nite + 'start']
        start_end_dict[file_id][speaker][sent_id]['end'] = child.attrib[nite + 'end']
        idx += 1

    idx_dict[speaker] = idx

    f_fixed.write('_'.join([file_id, speaker, sent_id]) + '\n')


def get_start_end_time_of_sentence(ids_dir, ids_fixed_dir, nxt_terminals_dir):
    start_end_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))

    for ids_file in Path(ids_dir).iterdir():
        f = open(ids_file)

        file_id = ids_file.stem
        if not os.path.exists(f'{nxt_terminals_dir}/{file_id}.A.terminals.xml'):
            print('no nxt annotation')
            print(file_id)
            print()
            continue

        f_fixed = open(f'{ids_fixed_dir}/{file_id}.ids', mode='w')

        tree_dict = {
            A: ET.parse(f'{nxt_terminals_dir}/{file_id}.A.terminals.xml').getroot(),
            B: ET.parse(f'{nxt_terminals_dir}/{file_id}.B.terminals.xml').getroot()
        }
        idx_dict = {'A': 0, 'B': 0}

        for id_line in f:
            update_start_end_dict(file_id, id_line, tree_dict, idx_dict, start_end_dict, f_fixed)

    return start_end_dict


def find_audio_file_path(file_id, swbd_dir):
    swbd_subdir = ['swb1_d1', 'swb1_d2', 'swb1_d3', 'swb1_d4']
    for sub in swbd_subdir:
        audio_file_name = file_id[:2] + '0' + file_id[2:] + '.sph'
        audio_file_path = f'{swbd_dir}/{sub}/data/{audio_file_name}'
        if os.path.isfile(audio_file_path):
            return audio_file_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nxt_terminals_dir',
                        help='directory of terminals annotation of NXT')
    parser.add_argument('--treebank3_converted_dir',
                        help='directory of converted treebank3')
    parser.add_argument('--swbd_audio_dir',
                        help='switchboard audio dir (swb1_LDC97S62)')
    parser.add_argument('--swbd_trim_dir',
                        help='output directory (trimmed swbd audio files)')
    args = parser.parse_args()

    ids_dir = f'{args.treebank3_converted_dir}/ids'
    ids_fixed_dir = f'{args.treebank3_converted_dir}/ids_fixed'

    os.makedirs(ids_fixed_dir, exist_ok=True)

    start_end_dict = get_start_end_time_of_sentence(ids_dir, ids_fixed_dir, args.nxt_terminals_dir)

    # trimming clips
    for file_id, annotations in start_end_dict.items():
        os.makedirs(f'{args.swbd_trim_dir}/{file_id}', exist_ok=True)
        audio_file_path = find_audio_file_path(file_id, args.swbd_audio_dir)
        start_end_A = annotations[A]
        start_end_B = annotations[B]
        trim_clip(start_end_A, A, args.swbd_trim_dir, audio_file_path)
        trim_clip(start_end_B, B, args.swbd_trim_dir, audio_file_path)
