import re
import conll18.conll18_ud_eval as conll18
from argparse import ArgumentParser
import xml.etree.ElementTree as ET


class CoNLLUToken:
    def __init__(self, id, form, upos, head, deprel, lemma=None, xpos=None, feats=None, deps=None, misc=None):
        self.id = id
        self.form = form
        self.lemma = lemma
        self.upos = upos
        self.xpos = xpos
        self.feats = feats
        self.head = head
        self.deprel = deprel
        self.deps = deps
        self.misc = misc

    def convert_to_line(self):
        '''
        Return conllu formatted line (str)

        >>> conllu = CoNLLUToken(1, "I", "NOUN", 2, "nsubj")
        >>> conllu.convert_to_line().split()
        ['1', 'I', '_', 'NOUN', '_', '_', '2', 'nsubj', '_', '_']
        >>> conllu = CoNLLUToken(1, "", "NOUN", 2, "nsubj")
        >>> conllu.convert_to_line().split()
        ['1', '_', '_', 'NOUN', '_', '_', '2', 'nsubj', '_', '_']
        '''
        return "\t".join([
            f"{str(self.id) or '_'}",
            f"{self.form or '_'}",
            f"{self.lemma or '_'}",
            f"{self.upos or '_'}",
            f"{self.xpos or '_'}",
            f"{self.feats or '_'}",
            f"{str(self.head) or '_'}",
            f"{self.deprel or '_'}",
            f"{self.deps or '_'}",
            f"{self.misc or '_'}"
        ])


class CoNLLUTree:
    def __init__(self, sent_id, conllu_list: list[CoNLLUToken]):
        self.sent_id = sent_id
        self.conllu_list = conllu_list

    def find_root(self):
        for conllu in self.conllu_list:
            if conllu.head == 0:
                return conllu
        return None

    def has_root(self):
        for conllu in self.conllu_list:
            if conllu.head == 0:
                return True
        return False

    def has_multiple_roots(self):
        roots = 0
        for conllu in self.conllu_list:
            if conllu.head == 0:
                roots += 1
        return roots > 1

    def assign_root_deprel(self):
        for conllu in self.conllu_list:
            if conllu.deprel == 'root':
                conllu.head = 0

    def assign_root_index_one(self):
        self.conllu_list[0].head = 0
        self.conllu_list[0].deprel = 'root'

    def choose_root_from_multiple_candidates(self):
        chosen_token_id = -1
        for conllu in self.conllu_list:
            if conllu.head == 0:
                if chosen_token_id == -1:
                    chosen_token_id = conllu.id
                    continue
                conllu.head = chosen_token_id

    def assign_headless_words_to_root(self):
        root = self.find_root()
        assert root is not None
        for conllu in self.conllu_list:
            if conllu.head == -1:
                conllu.head = root.id

    def find_cycles(self):
        still_cycle = True
        while still_cycle:
            still_cycle = False
            for conllu in self.conllu_list:
                focus_conllu = self.conllu_list[conllu.head - 1]
                seen = [conllu]
                while not focus_conllu.head == 0:
                    if focus_conllu not in seen:
                        seen.append(focus_conllu)
                        focus_conllu = self.conllu_list[focus_conllu.head - 1]
                    else:
                        still_cycle = True
                        root = self.find_root()
                        assert root is not None
                        focus_conllu.head = root.id
                        break

    def post_processing(self):
        '''
        Post-processing for the dependency tree constraint (single-root and acyclic)
        same as https://aclanthology.org/N19-1077/

        >>> conllu_tree = CoNLLUTree('sent_id', \
            [ \
                CoNLLUToken(1, "C", "11", 2, "5"), \
                CoNLLUToken(2, "EST", "19", -1, "-1"), \
                CoNLLUToken(3, "C", "11", 5, "5"), \
                CoNLLUToken(4, "EST", "19", 5, "2"), \
                CoNLLUToken(5, "LIÉ", "4", 2, "10"), \
                CoNLLUToken(6, "QUOI", "1", 5, "13"), \
            ])
        >>> conllu_tree.post_processing()
        '''

        if not self.has_root():
            self.assign_root_deprel()
        if not self.has_root():
            self.assign_root_index_one()
        assert self.has_root()
        if self.has_multiple_roots():
            self.choose_root_from_multiple_candidates()
        assert self.has_root()
        self.assign_headless_words_to_root()
        self.find_cycles()

    def convert_to_lines(self) -> str:
        lines = f"# sent_id = {self.sent_id}\n"
        for t in self.conllu_list:
            lines += t.convert_to_line() + "\n"
        return lines


class TreeEvaluator:
    '''
    Evaluator for the CTC output.
    calculate following metrics against gold tree: WER, CER, POS, UAS, LAS
    '''

    def __init__(self, pos_encoder=None, dep_encoder=None):
        self.pos_encoder = pos_encoder
        self.dep_encoder = dep_encoder
        self.conllu_trees: dict[str, CoNLLUTree] = {}

    def decode_one_token(self, token, idx, length):
        '''
        Return tuple of form and tag ids (POS, head_relpos, DEP).
        If a correspong tag isn't found, tag id is -1.
        If there are multiple tags, the first one is extracted.
        If the head is out of token range, regard it as "root" or the last token.

        >>> token = "I<POS12><R2><DEP24>"
        >>> TreeEvaluator().decode_one_token(token, 1, 4)
        {'form': 'I', 'POS': 12, 'head': 3, 'DEP': 24}
        >>> token = "have<R2><L3><POS0><DEP10><POS3>"
        >>> TreeEvaluator().decode_one_token(token, 2, 4)
        {'form': 'have', 'POS': 0, 'head': 4, 'DEP': 10}
        >>> token = "have<POS3>"
        >>> TreeEvaluator().decode_one_token(token, 2, 4)
        {'form': 'have', 'POS': 3, 'head': -1, 'DEP': -1}
        >>> token = "<POS0><R2><DEP24>"
        >>> TreeEvaluator().decode_one_token(token, 1, 4)
        {'form': '_', 'POS': 0, 'head': 3, 'DEP': 24}
        >>> token = "a"
        >>> TreeEvaluator().decode_one_token(token, 1, 4)
        {'form': 'a', 'POS': -1, 'head': -1, 'DEP': -1}
        >>> token = ""
        >>> TreeEvaluator().decode_one_token(token, 1, 4)
        {'form': '_', 'POS': -1, 'head': -1, 'DEP': -1}
        '''

        form_idx = 0
        for i, c in enumerate(token, 1):
            if c == "<":
                break
            form_idx = i
        form = token[:form_idx] or '_'
        pos_id = -1
        head_id = -1
        dep_id = -1
        tags = re.findall(r'<([A-Z]+)(\d+)>', token[form_idx:])
        for (tag, tag_id) in tags:
            if tag == 'POS' and pos_id == -1:
                pos_id = int(tag_id)
            elif tag == 'L' and head_id == -1:
                # heuristics: if head is less than 0, regard it as 0 (root)
                head_id = max(idx - int(tag_id), 0)
            elif tag == 'R' and head_id == -1:
                # heuristics: if head is greater than length, regard it as the last token
                head_id = min(idx + int(tag_id), length)
            elif tag == 'DEP' and dep_id == -1:
                dep_id = int(tag_id)
        return {"form": form, "POS": pos_id, "head": head_id, "DEP": dep_id}

    def decode_one_seq(self, sent_id: str, predicted_seq: str):
        '''
        Decode the predicted sequence into a dependency tree.
        If POS/DEP tag is missing, assign `X`/`dep` respectively.

        sent_id: sentence id
        predicted_seq : a predicted sequence formatted like "<s> I<POSi><Rj><DEPk> have...</s>"

        Returns: CoNLLUTree object (conllu representation of predicted_seq)
        '''

        # TODO: validation 入れる？

        conllu_list = []
        predicted_seq = predicted_seq.replace("<s>", "").replace("</s>", "").strip()
        tokens = predicted_seq.split(' ')
        for i, t in enumerate(tokens, 1):
            token_dict = self.decode_one_token(t, i, len(tokens))
            conllu_list.append(CoNLLUToken(
                i,
                token_dict['form'],
                self.pos_encoder.ind2lab.get(token_dict['POS'], 'X'),
                token_dict['head'],
                self.dep_encoder.ind2lab.get(token_dict['DEP'], 'dep')
            ))
        return CoNLLUTree(sent_id, conllu_list)

    def decode(self, ids, predicted_seqs):
        for sent_id, seq in zip(ids, predicted_seqs):
            conllu_tree = self.decode_one_seq(sent_id, seq)
            conllu_tree.post_processing()
            self.conllu_trees[sent_id] = conllu_tree

    def write_to_file(self, path, sent_ids: list[str]):
        with open(path, 'w') as f:
            for sent_id in sent_ids:
                f.write(self.conllu_trees[sent_id].convert_to_lines())
                f.write("\n")

    def write_trans_to_file(self, path, sent_ids: list[str]):
        with open(path, 'w') as f:
            for sent_id in sent_ids:
                trans_line = ''
                for conllu in self.conllu_trees[sent_id].conllu_list:
                    trans_line += conllu.form.lower() + ' '
                trans_line += '(' + sent_id + ')' + '\n'
                f.write(trans_line)

    def sort_sgml(self, sgml_path: str, output_path: str):
        '''
        Sort sgml file according to "sequence" keys.
        This is necessary particularly for swbd corpus because
        the ordering of output sgml file is not aligned, which affects
        UPOS/UAS/LAS evaluation.
        '''

        def _sortchildrenby(parent, attr):
            parent[:] = sorted(parent, key=lambda child: child.get(attr))

        with open(sgml_path) as f:
            del_idx = []
            lines = f.readlines()
            for i in range(len(lines)):
                line = lines[i]
                if line.startswith("<SPEAKER") or line.startswith("</SPEAKER"):
                    del_idx.append(i)
            for j, idx in enumerate(del_idx):
                del lines[idx - j]
        root = ET.fromstringlist(lines)
        for c in root.iter("PATH"):
            c.attrib["sequence"] = int(c.attrib["sequence"])
        _sortchildrenby(root, "sequence")
        for c in root.iter("PATH"):
            c.attrib["sequence"] = str(c.attrib["sequence"])
        tree = ET.ElementTree(root)
        tree.write(output_path)

    def evaluate_conllu(self, gold_path, predicted_path, path_sgml, analysis=False):
        subparser = ArgumentParser()
        subparser.add_argument("gold_file", type=str,
                               help="Name of the CoNLL-U file with the gold data.")
        subparser.add_argument("system_file", type=str,
                               help="Name of the CoNLL-U file with the predicted data.")
        subparser.add_argument("sgml_file", type=str,
                               help="Path of the output of SCLITE with sgml format.")
        subparser.add_argument("--verbose", "-v", default=False, action="store_true",
                               help="Print all metrics.")
        subparser.add_argument("--counts", "-c", default=False, action="store_true",
                               help="Print raw counts of correct/gold/system/aligned words instead of prec/rec/F1 for all metrics.")

        self.sort_sgml(path_sgml, path_sgml + "_sorted")
        subargs = subparser.parse_args([gold_path, predicted_path, path_sgml + "_sorted"])

        # Evaluate
        evaluation, pos_stat, uas_list = conll18.evaluate_wrapper(subargs, analysis)
        uas = 100 * evaluation["UAS"].f1
        las = 100 * evaluation["LAS"].f1

        print(repr(round(uas, 2)) + "\t" + repr(round(las, 2)))
        return evaluation, pos_stat, uas_list
