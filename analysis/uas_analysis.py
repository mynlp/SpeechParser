import csv
import xml.etree.ElementTree as ET
import pyconll

# !!Change variables here!!
textless_dir = './textless_dir'
wav2tree_dir = './wav2tree_dir'
gold_conllu_file = './gold/test.conllu'
textless_result_csv = './textless_wins.csv'
wav2tree_result_csv = './wav2tree_wins.csv'


textless_tree = ET.parse(textless_dir + '/test_predicted.conllu_trans.sgml_sorted')
textless_root = textless_tree.getroot()
wav2tree_tree = ET.parse(wav2tree_dir + '/test_predicted.conllu_trans.sgml_sorted')
wav2tree_root = wav2tree_tree.getroot()
gold_corpus = pyconll.load_from_file(gold_conllu_file)
textless_corpus = pyconll.load_from_file(f'{textless_dir}/test_predicted.conllu')
wav2tree_corpus = pyconll.load_from_file(f'{wav2tree_dir}/test_predicted.conllu')
textless_uas = f'{textless_dir}/analysis/uas.txt'
wav2tree_uas = f'{wav2tree_dir}/analysis/uas.txt'

textless_uas_list, wav2tree_uas_list = [], []
with open(textless_uas) as t, open(wav2tree_uas) as w:
    for t_uas, w_uas in zip(t, w):
        textless_uas_list.append(float(t_uas))
        wav2tree_uas_list.append(float(w_uas))

textless_uas_avg = sum(textless_uas_list) / len(textless_uas_list)
wav2tree_uas_avg = sum(wav2tree_uas_list) / len(wav2tree_uas_list)


def extract_types(alignment):
    return [c[0] for c in alignment.split(':')]


def no_wer(types):
    for t in types:
        if t != 'C':
            return False
    return True


def textless_wins(t_uas, w_uas):
    return t_uas == 1.0 and 100 * w_uas <= wav2tree_uas_avg


def wav2tree_wins(t_uas, w_uas):
    return w_uas == 1.0 and 100 * t_uas <= textless_uas_avg


def remove_comments(input_string):
    lines = input_string.split('\n')
    result_lines = [line for line in lines if not line.strip().startswith('#')]
    result_string = '\n'.join(result_lines)
    return result_string


def generate_csv_line(t_uas, w_uas, g_sent, t_sent, w_sent):
    return [
                g_sent.id,
                t_uas,
                w_uas,
                g_sent.text, 
                remove_comments(g_sent.conll()),
                remove_comments(t_sent.conll()),
                remove_comments(w_sent.conll())
            ]


def is_correct(gold_sent, pred_sent):
    for g_token, p_token in zip(gold_sent, pred_sent):
        if g_token.head != p_token.head:
            return False
    return True


with open(textless_uas) as t, open(wav2tree_uas) as w, open(textless_result_csv, 'w', newline='') as tcsv, open(wav2tree_result_csv, 'w', newline='') as wcsv:
    tcsv_writer = csv.writer(tcsv)
    wcsv_writer = csv.writer(wcsv)
    tcsv_writer.writerow(['ID', 'textless_UAS', 'wav2tree_UAS', 'text', 'gold_conllu', 'textless_conllu', 'wav2tree_conllu'])
    wcsv_writer.writerow(['ID', 'textless_UAS', 'wav2tree_UAS', 'text', 'gold_conllu', 'textless_conllu', 'wav2tree_conllu'])
    for t_uas, w_uas, g_sent, t_sent, w_sent, t_align, w_align in zip(t, w, gold_corpus, textless_corpus, wav2tree_corpus, textless_root, wav2tree_root):
        # ad-hoc implementation. to be fixed!!
        if is_correct(g_sent, t_sent) and is_correct(g_sent, w_sent):
            continue

        types_textless = extract_types(t_align.text.lstrip())
        types_wav2tree = extract_types(w_align.text.lstrip())
        if (not no_wer(types_textless)) or (not no_wer(types_wav2tree)):
            continue
 
        t_uas, w_uas = float(t_uas.strip()), float(w_uas.strip())
        if textless_wins(t_uas, w_uas):
            if 4 <= len(g_sent):
                tcsv_writer.writerow(generate_csv_line(str(round(t_uas, 2)), str(round(w_uas, 2)), g_sent, t_sent, w_sent))
        if wav2tree_wins(t_uas, w_uas):
            if 4 <= len(g_sent):
                wcsv_writer.writerow(generate_csv_line(str(round(t_uas, 2)), str(round(w_uas, 2)), g_sent, t_sent, w_sent))
