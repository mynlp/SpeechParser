"""
Author
-----
Shunsuke KANDO
"""

# Extract corresponding conllus from csv dataset

import argparse
import csv
import unicodedata


def split_id(ID: str) -> (str, str, str):
    '''
    Return the name of corpus, corpus_sub, and its id number

    >>> split_id('cefc-cfpp-Blanche_Duchemin_F_25_Reine_Ceret_F_60_11e-1137')
    ['cfpp', 'Blanche_Duchemin_F_25_Reine_Ceret_F_60_11e', '1137']
    >>> split_id('cefc-reunions-de-travail-OF1_SeanceTravail_4dec07-2155')
    ['reunions-de-travail', 'OF1_SeanceTravail_4dec07', '2155']
    >>> split_id('cefc-frenchoralnarrative-Mastre_106-1_CONTE_DE_BOURGOGNE_1-93')
    ['frenchoralnarrative', 'Mastre_106-1_CONTE_DE_BOURGOGNE_1', '93']
    >>> split_id('cefc-cfpb-1200-2-679')
    ['cfpb', 'CFPB-1200-2', '679']
    '''
    cefc_removed = ID.split('-', 1)[1]
    rest_split = cefc_removed.split('-')
    if cefc_removed.startswith('reunions-de-travail'):
        return ['reunions-de-travail', '-'.join(rest_split[3:-1]), rest_split[-1]]
    if cefc_removed.startswith('cfpb'):
        return [rest_split[0], 'CFPB-' + '-'.join(rest_split[1:-1]), rest_split[-1]]
    else:
        return [rest_split[0], '-'.join(rest_split[1:-1]), rest_split[-1]]


def extract_conllu_and_trans(sent_id: str, path_to_tree: str) -> (str, str):
    conllu_lines = ''
    trans_line = ''
    with open(path_to_tree, 'r') as f:
        found = False
        i = 0
        for l in f:
            if l == f'# sent_id = {sent_id}\n':
                conllu_lines += l
                found = True
                continue
            if found and i == 0:
                conllu_lines += l
                i += 1
                continue
            if found and l == '\n':
                break
            if not found:
                continue
            features = l.split('\t')
            conllu_lines += '\t'.join(features[:10]) + '\n'
            # if features[1] == '#' or ' ' in features[1]:
            #     return None, None
            form = "".join(filter(lambda c: unicodedata.category(c) != "Zs", features[1])).lower()
            trans_line += form + ' '
            i += 1
        trans_line += '(' + sent_id + ')'
    return conllu_lines, trans_line


def write_gold_conllu_and_trans(source_path, conllu_path, tree_dir):
    trans_path = conllu_path + '_trans'
    with open(source_path) as fin, open(conllu_path, 'w') as fconllu, open(trans_path, 'w') as ftrans:
        dict_reader = csv.DictReader(fin)
        for row in dict_reader:
            sent_id = row['ID']
            corpus, corpus_sub, id_num = split_id(sent_id)
            path_to_tree = f'{tree_dir}/{corpus}/{corpus_sub}.orfeo'
            conllu_lines, trans_line = extract_conllu_and_trans(sent_id, path_to_tree)
            if conllu_lines is None:
                continue
            fconllu.write(conllu_lines)
            fconllu.write('\n')
            ftrans.write(trans_line)
            ftrans.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # TODO: remove default
    parser.add_argument('--csv_dir',
                        help='directory of orfeo dataset (in csv)')
    parser.add_argument('--out_dir',
                        help='directory to output conllus')
    parser.add_argument('--tree_dir',
                        help='directory of original conllu files of orfeo treebank')
    args = parser.parse_args()

    for file_name in ['dev', 'test']:
        source_path = f'{args.csv_dir}/{file_name}.csv'
        conllu_path = f'{args.out_dir}/{file_name}.conllu'
        write_gold_conllu_and_trans(source_path, conllu_path, args.tree_dir)
