#!/usr/bin/env python3

# Compatible with Python 2.7 and 3.2+, can be used either as a module
# or a standalone executable.
#
# Copyright 2017, 2018 Institute of Formal and Applied Linguistics (UFAL),
# Faculty of Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Authors: Milan Straka, Martin Popel <surname@ufal.mff.cuni.cz>
# Modifications by Shunsuke Kando
#
# Changelog:
# - [12 Apr 2018] Version 0.9: Initial release.
# - [19 Apr 2018] Version 1.0: Fix bug in MLAS (duplicate entries in functional_children).
#                              Add --counts option.
# - [02 May 2018] Version 1.1: When removing spaces to match gold and system characters,
#                              consider all Unicode characters of category Zs instead of
#                              just ASCII space.
# - [25 Jun 2018] Version 1.2: Use python3 in the she-bang (instead of python).
#                              In Python2, make the whole computation use `unicode` strings.

# Command line usage
# ------------------
# conll18_ud_eval.py [-v] gold_conllu_file system_conllu_file
#
# - if no -v is given, only the official CoNLL18 UD Shared Task evaluation metrics
#   are printed
# - if -v is given, more metrics are printed (as precision, recall, F1 score,
#   and in case the metric is computed on aligned words also accuracy on these):
#   - Tokens: how well do the gold tokens match system tokens
#   - Sentences: how well do the gold sentences match system sentences
#   - Words: how well can the gold words be aligned to system words
#   - UPOS: using aligned words, how well does UPOS match
#   - XPOS: using aligned words, how well does XPOS match
#   - UFeats: using aligned words, how well does universal FEATS match
#   - AllTags: using aligned words, how well does UPOS+XPOS+FEATS match
#   - Lemmas: using aligned words, how well does LEMMA match
#   - UAS: using aligned words, how well does HEAD match
#   - LAS: using aligned words, how well does HEAD+DEPREL(ignoring subtypes) match
#   - CLAS: using aligned words with content DEPREL, how well does
#       HEAD+DEPREL(ignoring subtypes) match
#   - MLAS: using aligned words with content DEPREL, how well does
#       HEAD+DEPREL(ignoring subtypes)+UPOS+UFEATS+FunctionalChildren(DEPREL+UPOS+UFEATS) match
#   - BLEX: using aligned words with content DEPREL, how well does
#       HEAD+DEPREL(ignoring subtypes)+LEMMAS match
# - if -c is given, raw counts of correct/gold_total/system_total/aligned words are printed
#   instead of precision/recall/F1/AlignedAccuracy for all metrics.

# API usage
# ---------
# - load_conllu(file)
#   - loads CoNLL-U file from given file object to an internal representation
#   - the file object should return str in both Python 2 and Python 3
#   - raises UDError exception if the given file cannot be loaded
# - evaluate(gold_ud, system_ud)
#   - evaluate the given gold and system CoNLL-U files (loaded with load_conllu)
#   - raises UDError if the concatenated tokens of gold and system file do not match
#   - returns a dictionary with the metrics described above, each metric having
#     three fields: precision, recall and f1

# Description of token matching
# -----------------------------
# In order to match tokens of gold file and system file, we consider the text
# resulting from concatenation of gold tokens and text resulting from
# concatenation of system tokens. These texts should match -- if they do not,
# the evaluation fails. =
#
# If the texts do match, every token is represented as a range in this original
# text, and tokens are equal only if their range is the same.

# CHANGE FOR ASR : (2022, WAV2TREE)
#  The description above is the heart of the problem for our ASR output.
#  We use the SCLITE alignment tools to add dummy tokens in the system list (the ASR Input) when there is a deletion
#  In the case of insertion, we add dummy tokens in the gold list (the gold input).
#  The insertion of dummy tokens allow us to mitigate the alignment problem and keep the corresponding input aligned in each list
#  Thus, in this implementation any insertion or deletion count as 1 error and do not propagate.


# Description of word matching
# ----------------------------
# When matching words of gold file and system file, we first match the tokens.
# The words which are also tokens are matched as tokens, but words in multi-word
# tokens have to be handled differently.
#
# To handle multi-word tokens, we start by finding "multi-word spans".
# Multi-word span is a span in the original text such that
# - it contains at least one multi-word token
# - all multi-word tokens in the span (considering both gold and system ones)
#   are completely inside the span (i.e., they do not "stick out")
# - the multi-word span is as small as possible
#
# For every multi-word span, we align the gold and system words completely
# inside this span using LCS on their FORMs. The words not intersecting
# (even partially) any multi-word span are then aligned as tokens.


from __future__ import division
from __future__ import print_function

import argparse
import io
import sys
import unicodedata
import Levenshtein
import unittest
from tqdm import tqdm
from collections import defaultdict

# CoNLL-U column names
ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC = range(10)

# Content and functional relations
CONTENT_DEPRELS = {
    "nsubj", "obj", "iobj", "csubj", "ccomp", "xcomp", "obl", "vocative",
    "expl", "dislocated", "advcl", "advmod", "discourse", "nmod", "appos",
    "nummod", "acl", "amod", "conj", "fixed", "flat", "compound", "list",
    "parataxis", "orphan", "goeswith", "reparandum", "root", "dep"
}

FUNCTIONAL_DEPRELS = {
    "aux", "cop", "mark", "det", "clf", "case", "cc"
}

UNIVERSAL_FEATURES = {
    "PronType", "NumType", "Poss", "Reflex", "Foreign", "Abbr", "Gender",
    "Animacy", "Number", "Case", "Definite", "Degree", "VerbForm", "Mood",
    "Tense", "Aspect", "Voice", "Evident", "Polarity", "Person", "Polite"
}


# UD Error is used when raising exceptions in this module
class UDError(Exception):
    pass


# Conversion methods handling `str` <-> `unicode` conversions in Python2
def _decode(text):
    return text if sys.version_info[0] >= 3 or not isinstance(text, str) else text.decode("utf-8")


def _encode(text):
    return text if sys.version_info[0] >= 3 or not isinstance(text, unicode) else text.encode("utf-8")


class UDRepresentation:
    def __init__(self):
        # Characters of all the tokens in the whole file.
        # Whitespace between tokens is not included.
        self.characters = []
        # List of UDSpan instances with start&end indices into `characters`.
        self.tokens = []
        # List of UDWord instances.
        self.words = []
        # List of UDSpan instances with start&end indices into `characters`.
        self.sentences = []


class UDSpan:
    def __init__(self, start, end):
        self.start = start
        # Note that self.end marks the first position **after the end** of span,
        # so we can use characters[start:end] or range(start, end).
        self.end = end


class UDWord:
    def __init__(self, span, columns, is_multiword):
        # Span of this word (or MWT, see below) within ud_representation.characters.
        self.span = span
        # 10 columns of the CoNLL-U file: ID, FORM, LEMMA,...
        self.columns = columns
        # is_multiword==True means that this word is part of a multi-word token.
        # In that case, self.span marks the span of the whole multi-word token.
        self.is_multiword = is_multiword
        # Reference to the UDWord instance representing the HEAD (or None if root).
        self.parent = None
        # List of references to UDWord instances representing functional-deprel children.
        self.functional_children = []
        # Only consider universal FEATS.
        self.columns[FEATS] = "|".join(sorted(feat for feat in columns[FEATS].split("|")
                                              if feat.split("=", 1)[0] in UNIVERSAL_FEATURES))
        # Let's ignore language-specific deprel subtypes.
        self.columns[DEPREL] = columns[DEPREL].split(":")[0]
        # Precompute which deprels are CONTENT_DEPRELS and which FUNCTIONAL_DEPRELS
        self.is_content_deprel = self.columns[DEPREL] in CONTENT_DEPRELS
        self.is_functional_deprel = self.columns[DEPREL] in FUNCTIONAL_DEPRELS


class Score:
    def __init__(self, gold_total, system_total, correct, aligned_total=None):
        self.correct = correct
        self.gold_total = gold_total
        self.system_total = system_total
        self.aligned_total = aligned_total
        self.precision = correct / system_total if system_total else 0.0
        self.recall = correct / gold_total if gold_total else 0.0
        self.f1 = 2 * correct / (system_total + gold_total) if system_total + gold_total else 0.0
        self.aligned_accuracy = correct / aligned_total if aligned_total else aligned_total


class AlignmentWord:
    def __init__(self, gold_word, system_word):
        self.gold_word = gold_word
        self.system_word = system_word


class Alignment:
    def __init__(self, gold_words, system_words):
        self.gold_words = gold_words
        self.system_words = system_words
        self.matched_words = []
        self.matched_words_map = {}

    def append_aligned_words(self, gold_word, system_word):
        self.matched_words.append(AlignmentWord(gold_word, system_word))
        self.matched_words_map[system_word] = gold_word


class SmglToken:
    def __init__(self, type, ref, hyp):
        self.type = type
        self.ref = ref
        self.hyp = hyp


def load_conllu_file(path):
    _file = open(path, mode="r", **({"encoding": "utf-8"} if sys.version_info >= (3, 0) else {}))
    return load_conllu(_file)


def load_smgl_file(path):
    _file = open(path, mode="r", **({"encoding": "utf-8"} if sys.version_info >= (3, 0) else {}))
    return load_smgl(_file)


def load_smgl(file):
    aligned_token_smgl = []
    for line in file:
        if line.startswith("<"):
            continue
        field = line.split(":")  # create a list of string of form "type,ref,hyp"
        for token in field:
            tokenelem = token.split(",")
            tok = SmglToken(tokenelem[0], tokenelem[1], tokenelem[2])
            aligned_token_smgl.append(tok)
    return aligned_token_smgl


# Load given CoNLL-U file into internal representation
def load_conllu(file):
    # Internal representation classes
    ud = UDRepresentation()
    # Load the CoNLL-U file
    index, sentence_start = 0, None
    while True:
        line = file.readline().upper()
        if not line:
            break
        line = _decode(line.rstrip("\r\n"))

        # Handle sentence start boundaries
        if sentence_start is None:
            # Skip comments
            if line.startswith("#"):
                continue
            # Start a new sentence
            ud.sentences.append(UDSpan(index, 0))
            sentence_start = len(ud.words)

        if not line:
            # Add parent and children UDWord links and check there are no cycles
            def process_word(word):
                if word.parent == "remapping":
                    raise UDError("There is a cycle in a sentence")
                if word.parent is None:
                    head = int(word.columns[HEAD])
                    if head < 0 or head > len(ud.words) - sentence_start:
                        raise UDError("HEAD '{}' points outside of the sentence".format(_encode(word.columns[HEAD])))
                    if head:
                        parent = ud.words[sentence_start + head - 1]
                        word.parent = "remapping"
                        process_word(parent)
                        word.parent = parent

            for word in ud.words[sentence_start:]:
                try:
                    process_word(word)
                except UDError:  # stop processing this line, cyle = invalid, CUSTOM
                    print("cycle found, ignoring sentence")
                    print(word.columns)
                    continue
            # func_children cannot be assigned within process_word
            # because it is called recursively and may result in adding one child twice.
            for word in ud.words[sentence_start:]:
                if word.parent and word.is_functional_deprel:
                    word.parent.functional_children.append(word)

            # Check there is a single root node

            # if len([word for word in ud.words[sentence_start:] if word.parent is None]) != 1:
            #   raise UDError("There are multiple roots in a sentence")

            # End the sentence
            ud.sentences[-1].end = index
            sentence_start = None
            continue

        # Read next token/word
        columns = line.split("\t")

        if len(columns) < 10:
            raise UDError(
                "The CoNLL-U line does not contain at least 10 tab-separated columns: '{}'".format(_encode(line)))

        # Skip empty nodes
        if "." in columns[ID]:
            continue

        # Delete spaces from FORM, so gold.characters == system.characters
        # even if one of them tokenizes the space. Use any Unicode character
        # with category Zs.
        columns[FORM] = "".join(filter(lambda c: unicodedata.category(c) != "Zs", columns[FORM]))
        if not columns[FORM]:
            raise UDError("There is an empty FORM in the CoNLL-U file")

        # Save token
        ud.characters.extend(columns[FORM])
        ud.tokens.append(UDSpan(index, index + len(columns[FORM])))
        index += len(columns[FORM])

        # Handle multi-word tokens to save word(s)
        if "-" in columns[ID]:
            try:
                start, end = map(int, columns[ID].split("-"))
            except:
                raise UDError("Cannot parse multi-word token ID '{}'".format(_encode(columns[ID])))

            for _ in range(start, end + 1):
                word_line = _decode(file.readline().rstrip("\r\n"))
                word_columns = word_line.split("\t")
                if len(word_columns) < 10:
                    raise UDError(
                        "The CoNLL-U line does not contain at least 10 tab-separated columns: '{}'".format(
                            _encode(word_line)))
                ud.words.append(UDWord(ud.tokens[-1], word_columns, is_multiword=True))
        # Basic tokens/words
        else:
            try:
                word_id = int(columns[ID])
            except:
                raise UDError("Cannot parse word ID '{}'".format(_encode(columns[ID])))
            if word_id != len(ud.words) - sentence_start + 1:
                raise UDError("Incorrect word ID '{}' for word '{}', expected '{}'".format(
                    _encode(columns[ID]), _encode(columns[FORM]), len(ud.words) - sentence_start + 1))

            try:
                head_id = int(columns[HEAD])
            except:
                raise UDError("Cannot parse HEAD '{}'".format(_encode(columns[HEAD])))
            if head_id < 0:
                raise UDError("HEAD cannot be negative")

            ud.words.append(UDWord(ud.tokens[-1], columns, is_multiword=False))

    if sentence_start is not None:
        raise UDError("The CoNLL-U file does not end with empty line")

    return ud


# Evaluate the gold and system treebanks (loaded using load_conllu).
class Evaluator():
    def __init__(self, gold_ud, system_ud, alignement_smgl):
        self.gold_ud = gold_ud
        self.system_ud = system_ud
        self.alignement_smgl = alignement_smgl

    def spans_score(self, gold_spans, system_spans):
        correct, gi, si = 0, 0, 0
        while gi < len(gold_spans) and si < len(system_spans):
            if system_spans[si].start < gold_spans[gi].start:
                si += 1
            elif gold_spans[gi].start < system_spans[si].start:
                gi += 1
            else:
                correct += gold_spans[gi].end == system_spans[si].end
                si += 1
                gi += 1

        return Score(len(gold_spans), len(system_spans), correct)

    # faster matching library than difflibs. Difflib is quadratics, levenstein is m*n.
    def string_similarity(self, str1, str2):
        return Levenshtein.ratio(str1, str2)

    def _is_token_in_sent(self, sent_spans, token_spans):
        return sent_spans.start <= token_spans.start and sent_spans.end >= token_spans.end

    def get_token_in_sent(self, sent_spans, token_spans, lastIndex):
        tokenList = []
        index = lastIndex
        for i in range(lastIndex, len(token_spans)):
            if self._is_token_in_sent(sent_spans, token_spans[i]):
                tokenList.append(token_spans[i])
            else:
                index = i
                break
        return tokenList, index

    def spans_score_unaligned_tokens(self, sentences_gold_spans, sentences_system_spans, token_gold_spans,
                                     token_system_spans,
                                     gold_characters, system_characters):
        correct_sent = 0
        correct_tokens = 0
        nb_token_diff = 0
        index_gold = index_sys = 0
        for s_g_spans, s_s_spans in zip(sentences_gold_spans, sentences_system_spans):
            # len_s_g=s_g_spans.start-s_g_spans.end
            # len_s_s=s_s_spans.start-s_s_spans.end
            string_gold = "".join(gold_characters[s_g_spans.start:s_g_spans.end])
            string_sys = "".join(system_characters[s_s_spans.start:s_s_spans.end])
            correct_sent += self.string_similarity(string_gold, string_sys)
            gold_tokens, index_gold = self.get_token_in_sent(s_g_spans, token_gold_spans, index_gold)
            sys_tokens, index_sys = self.get_token_in_sent(s_s_spans, token_system_spans, index_sys)
            if len(gold_tokens) != len(
                    sys_tokens):  # Need to realign token because of segmentation error in system tokens
                # assumption : system form should correspond to the gold token closest in form
                nb_token_diff += abs(len(gold_tokens) - len(sys_tokens))
                for t_s in sys_tokens:
                    max_sim = max([self.string_similarity("".join(gold_characters[t_g.start:t_g.end]),
                                                          "".join(system_characters[t_s.start:t_s.end]))
                                   for t_g in gold_tokens])
                    correct_tokens += max_sim

            else:  # proceed normaly
                correct_tokens += sum([self.string_similarity("".join(gold_characters[t_g.start:t_g.end]),
                                                              "".join(system_characters[t_s.start:t_s.end]))
                                       for t_g, t_s in zip(gold_tokens, sys_tokens)])
        return Score(len(sentences_gold_spans), len(sentences_system_spans), correct_sent), \
            Score(len(token_gold_spans), len(token_system_spans), correct_tokens), \
            Score(len(token_gold_spans), len(token_system_spans), nb_token_diff)

    def alignment_score(self, alignment, key_fn=None, filter_fn=None):
        if filter_fn is not None:
            gold = sum(1 for gold in alignment.gold_words if filter_fn(gold))
            system = sum(1 for system in alignment.system_words if filter_fn(system))
            aligned = sum(1 for word in alignment.matched_words if filter_fn(word.gold_word))
        else:
            gold = len(alignment.gold_words)
            system = len(alignment.system_words)
            aligned = len(alignment.matched_words)

        if key_fn is None:
            # Return score for whole aligned words
            return Score(gold, system, aligned)

        def gold_aligned_gold(word):
            return word

        def gold_aligned_system(word):
            return alignment.matched_words_map.get(word, "NotAligned") if word is not None else None

        correct = 0
        for words in alignment.matched_words:
            if filter_fn is None or filter_fn(words.gold_word):
                if key_fn(words.gold_word, gold_aligned_gold) == key_fn(words.system_word, gold_aligned_system):
                    correct += 1

        return Score(gold, system, correct, aligned)

    def beyond_end(self, words, i, multiword_span_end):
        if i >= len(words):
            return True
        if words[i].is_multiword:
            return words[i].span.start >= multiword_span_end
        return words[i].span.end > multiword_span_end

    def extend_end(self, word, multiword_span_end):
        if word.is_multiword and word.span.end > multiword_span_end:
            return word.span.end
        return multiword_span_end

    def find_multiword_span(self, gold_words, system_words, gi, si):
        # We know gold_words[gi].is_multiword or system_words[si].is_multiword.
        # Find the start of the multiword span (gs, ss), so the multiword span is minimal.
        # Initialize multiword_span_end characters index.
        if gold_words[gi].is_multiword:
            multiword_span_end = gold_words[gi].span.end
            if not system_words[si].is_multiword and system_words[si].span.start < gold_words[gi].span.start:
                si += 1
        else:  # if system_words[si].is_multiword
            multiword_span_end = system_words[si].span.end
            if not gold_words[gi].is_multiword and gold_words[gi].span.start < system_words[si].span.start:
                gi += 1
        gs, ss = gi, si

        # Find the end of the multiword span
        # (so both gi and si are pointing to the word following the multiword span end).
        while not self.beyond_end(gold_words, gi, multiword_span_end) or \
                not self.beyond_end(system_words, si, multiword_span_end):
            if gi < len(gold_words) and (si >= len(system_words) or
                                         gold_words[gi].span.start <= system_words[si].span.start):
                multiword_span_end = self.extend_end(gold_words[gi], multiword_span_end)
                gi += 1
            else:
                multiword_span_end = self.extend_end(system_words[si], multiword_span_end)
                si += 1
        return gs, ss, gi, si

    def compute_lcs(self, gold_words, system_words, gi, si, gs, ss):
        lcs = [[0] * (si - ss) for i in range(gi - gs)]
        for g in reversed(range(gi - gs)):
            for s in reversed(range(si - ss)):
                if gold_words[gs + g].columns[FORM].lower() == system_words[ss + s].columns[FORM].lower():
                    lcs[g][s] = 1 + (lcs[g + 1][s + 1] if g + 1 < gi - gs and s + 1 < si - ss else 0)
                lcs[g][s] = max(lcs[g][s], lcs[g + 1][s] if g + 1 < gi - gs else 0)
                lcs[g][s] = max(lcs[g][s], lcs[g][s + 1] if s + 1 < si - ss else 0)
        return lcs

    def align_words(self, gold_words, system_words):
        alignment = Alignment(gold_words, system_words)

        gi, si = 0, 0
        while gi < len(gold_words) and si < len(system_words):
            if gold_words[gi].is_multiword or system_words[si].is_multiword:
                # A: Multi-word tokens => align via LCS within the whole "multiword span".
                gs, ss, gi, si = self.find_multiword_span(gold_words, system_words, gi, si)

                if si > ss and gi > gs:
                    lcs = self.compute_lcs(gold_words, system_words, gi, si, gs, ss)

                    # Store aligned words
                    s, g = 0, 0
                    while g < gi - gs and s < si - ss:
                        if gold_words[gs + g].columns[FORM].lower() == system_words[ss + s].columns[FORM].lower():
                            alignment.append_aligned_words(gold_words[gs + g], system_words[ss + s])
                            g += 1
                            s += 1
                        elif lcs[g][s] == (lcs[g + 1][s] if g + 1 < gi - gs else 0):
                            g += 1
                        else:
                            s += 1
            else:
                # B: No multi-word token => align according to spans.
                if (gold_words[gi].span.start, gold_words[gi].span.end) == (
                        system_words[si].span.start, system_words[si].span.end):
                    alignment.append_aligned_words(gold_words[gi], system_words[si])
                    gi += 1
                    si += 1
                elif gold_words[gi].span.start <= system_words[si].span.start:
                    gi += 1
                else:
                    si += 1

        return alignment

    def align_words_ASR(self, gold_words, system_words, smgl_alignment):
        '''
        Align the two list extracted from the file.
        Parameters
        ----------
        gold_words : the gold words list from the gold file
        system_words : the system words list from the system file (ASR)
        smgl_alignment the smgl alignment obtained from the Sclite tools

        Returns
        alignment : return an instance of Alignment with the two list aligned with the insertion of dummy tokens in
        the gold list when there is an ASR insertion and dummy tokens inserted in the system list when there is a
         ASR deletion.
        -------

        '''
        print(len(system_words))
        print(len(gold_words))
        count_C_S = 0
        count_D = 0
        count_I = 0
        s_len = len(system_words)
        g_len = len(gold_words)
        for i in range(len(smgl_alignment)):
            smgl_token = smgl_alignment[i]
            if smgl_token.type == "C" or smgl_token.type == "S":  # if correct or substitued token (misspeling,..) then alignement is good
                count_C_S += 1
                continue
            if smgl_token.type == "D":  # Missing token in pred, add dummy to keep alignment between gold and sys
                count_D += 1
                if i == 0:
                    index = 0
                else:
                    index = i - 1
                try:
                    dummy_UD = [int(system_words[index].columns[0]) + 1, "DUMMYS", "_", "DUMMYS", "_", "_", 0, "DUMMYS",
                                "_", "_"]
                    system_words.insert(i, UDWord(UDSpan(system_words[index].span.end, system_words[index].span.end),
                                                  dummy_UD, is_multiword=False))
                except IndexError:
                    print(len(system_words))
                    print(len(gold_words))
                    print(len(smgl_alignment))
                    print(index)
                    print(f"C_S {count_C_S}, D {count_D}, I {count_I}")
                    raise IndexError()
            if smgl_token.type == "I":  # Added token in pred, add dummy to keep alignment between gold and sys
                count_I += 1
                if i == 0:
                    index = 0
                else:
                    index = i - 1
                dummy_UD = [int(gold_words[index].columns[0]) + 1, "DUMMYG", "_", "DUMMYG", "_", "_", 0, "DUMMYG", "_",
                            "_"]
                gold_words.insert(i, UDWord(UDSpan(gold_words[index].span.end, gold_words[index].span.end), dummy_UD,
                                            is_multiword=False))
        if len(gold_words) != len(system_words):
            # print([w.columns[1] for w in gold_words])
            # print([w.columns[1] for w in system_words])
            raise UDError(
                f"Alignement failed, need to debug. len gold : {len(gold_words)}, len system {len(system_words)}."
                f"This can be caused by a second blank line being inserted between sentence in conllu file.")
        alignment = Alignment(gold_words, system_words)
        # todo : deal with MWE in this
        for gold_word, sys_word in zip(gold_words, system_words):
            alignment.append_aligned_words(gold_word, sys_word)

        return alignment

    def pos_analysis(self, alignment: Alignment):
        total_words = 0
        pos_stat = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))
        result_dir = "fr_wav2tree"

        for w in alignment.matched_words:
            gold_cols = w.gold_word.columns
            pred_cols = w.system_word.columns
            total_words += 1

            pos = gold_cols[UPOS]
            # ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC = range(10)
            if type(gold_cols[ID]) is str:
                gold_headpos = str(int(gold_cols[HEAD]) - int(gold_cols[ID]))
                gold_deprel = gold_cols[DEPREL]
                pos_stat[pos]['HEAD'][gold_headpos]['gold'] += 1
                pos_stat[pos]['DEPREL'][gold_deprel]['gold'] += 1

                if type(pred_cols[ID]) is not str:
                    continue

                pred_headpos = str(int(pred_cols[HEAD]) - int(pred_cols[ID]))
                pred_deprel = pred_cols[DEPREL]
                if pos == pred_cols[UPOS]:
                    if gold_headpos == pred_headpos:
                        pos_stat[pos]['HEAD'][pred_headpos]['pred'] += 1
                    if gold_deprel == pred_deprel:
                        pos_stat[pos]['DEPREL'][pred_deprel]['pred'] += 1

        return pos_stat

    def tree_analysis(self, alignment: Alignment):
        def gold_aligned_system(word):
            return alignment.matched_words_map.get(word, "NotAligned") if word is not None else None

        uas_list = []
        correct = 0
        system_total = 0
        gold_total = 0
        for i, w in enumerate(alignment.matched_words):
            gold_word = w.gold_word
            system_word = w.system_word
            if int(gold_word.columns[ID]) == 1 and i != 0:
                uas_list.append(2 * correct / (system_total + gold_total))
                correct = 0
                system_total = 0
                gold_total = 0
            if system_word.columns[FORM] != "DUMMY_S":
                system_total += 1
            if gold_word.columns[FORM] != "DUMMY_G":
                gold_total += 1
            if gold_word.parent == gold_aligned_system(system_word.parent):
                correct += 1
        return uas_list

    def evaluate(self, analysis=False):
        '''
        # Check that the underlying character sequences do match.
        if gold_ud.characters != system_ud.characters:
            index = 0
            while index < len(gold_ud.characters) and index < len(system_ud.characters) and \
                    gold_ud.characters[index] == system_ud.characters[index]:
                index += 1

            raise UDError(
                "The concatenation of tokens in gold file and in system file differ!\n" +
                "First 20 differing characters in gold file: '{}' and system file: '{}'".format(
                    "".join(map(_encode, gold_ud.characters[index:index + 20])),
                    "".join(map(_encode, system_ud.characters[index:index + 20]))
                )
            )
        '''
        # Align words
        # alignment = align_words(gold_ud.words, system_ud.words)
        alignment = self.align_words_ASR(self.gold_ud.words, self.system_ud.words, self.alignement_smgl)
        sentence_score, token_score, seg_error_rate = self.spans_score_unaligned_tokens(self.gold_ud.sentences,
                                                                                        self.system_ud.sentences,
                                                                                        self.gold_ud.tokens,
                                                                                        self.system_ud.tokens,
                                                                                        self.gold_ud.characters,
                                                                                        self.system_ud.characters)
        pos_stat = None
        uas_list = None
        if analysis:
            pos_stat = self.pos_analysis(alignment)
            uas_list = self.tree_analysis(alignment)

        # sentence_score, token_score, seg_error_rate = spans_score_unaligned_tokens_quick(gold_ud.sentences, system_ud.sentences,
        #                                                                           gold_ud.tokens,system_ud.tokens)
        print(sentence_score)
        print(token_score)
        print(seg_error_rate)
        # Compute the F1-scores
        return {
            "Tokens": token_score,  # spans_score(gold_ud.tokens, system_ud.tokens),
            "Sentences": sentence_score,  # spans_score(gold_ud.sentences, system_ud.sentences),
            "seg_error_rate": seg_error_rate,
            "Words": self.alignment_score(alignment),
            "UPOS": self.alignment_score(alignment, lambda w, _: w.columns[UPOS]),
            "XPOS": self.alignment_score(alignment, lambda w, _: w.columns[XPOS]),
            "UFeats": self.alignment_score(alignment, lambda w, _: w.columns[FEATS]),
            "AllTags": self.alignment_score(alignment,
                                            lambda w, _: (w.columns[UPOS], w.columns[XPOS], w.columns[FEATS])),
            "Lemmas": self.alignment_score(alignment,
                                           lambda w, ga: w.columns[LEMMA] if ga(w).columns[LEMMA] != "_" else "_"),
            "UAS": self.alignment_score(alignment, lambda w, ga: ga(w.parent)),
            "LAS": self.alignment_score(alignment, lambda w, ga: (ga(w.parent), w.columns[DEPREL])),
            "CLAS": self.alignment_score(alignment, lambda w, ga: (ga(w.parent), w.columns[DEPREL]),
                                         filter_fn=lambda w: w.is_content_deprel),
            "MLAS": self.alignment_score(alignment,
                                         lambda w, ga: (
                                             ga(w.parent), w.columns[DEPREL], w.columns[UPOS], w.columns[FEATS],
                                             [(ga(c), c.columns[DEPREL], c.columns[UPOS], c.columns[FEATS])
                                              for c in w.functional_children]),
                                         filter_fn=lambda w: w.is_content_deprel),
            "BLEX": self.alignment_score(alignment, lambda w, ga: (ga(w.parent), w.columns[DEPREL],
                                                                   w.columns[LEMMA] if ga(w).columns[
                                                                                           LEMMA] != "_" else "_"),
                                         filter_fn=lambda w: w.is_content_deprel),
        }, pos_stat, uas_list


def evaluate_wrapper(args, analysis=False):
    # Load CoNLL-U files
    gold_ud = load_conllu_file(args.gold_file)
    system_ud = load_conllu_file(args.system_file)
    trans_alignement = load_smgl_file(args.sgml_file)
    eval = Evaluator(gold_ud, system_ud, trans_alignement)
    return eval.evaluate(analysis)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("gold_file", type=str,
                        help="Name of the CoNLL-U file with the gold data.")
    parser.add_argument("system_file", type=str,
                        help="Name of the CoNLL-U file with the predicted data.")
    parser.add_argument("sgml_file", type=str,
                        help="Path of the output of SCLITE with sgml format.")
    parser.add_argument("--verbose", "-v", default=False, action="store_true",
                        help="Print all metrics.")
    parser.add_argument("--counts", "-c", default=False, action="store_true",
                        help="Print raw counts of correct/gold/system/aligned words instead of prec/rec/F1 for all metrics.")
    parser.add_argument("--analysis", "-c", default=False, action="store_true",
                        help="Do analysis")
    args = parser.parse_args()

    # Evaluate
    evaluation, pos_stat, uas_list = evaluate_wrapper(args, analysis=args.analysis)

    # Print the evaluation
    if not args.verbose and not args.counts:
        print("LAS F1 Score: {:.2f}".format(100 * evaluation["LAS"].f1))
        print("MLAS Score: {:.2f}".format(100 * evaluation["MLAS"].f1))
        print("BLEX Score: {:.2f}".format(100 * evaluation["BLEX"].f1))
    else:
        if args.counts:
            print("Metric     | Correct   |      Gold | Predicted | Aligned")
        else:
            print("Metric     | Precision |    Recall |  F1 Score | AligndAcc")
        print("-----------+-----------+-----------+-----------+-----------")
        for metric in ["Tokens", "Sentences", "Words", "UPOS", "XPOS", "UFeats", "AllTags", "Lemmas", "UAS", "LAS",
                       "CLAS", "MLAS", "BLEX"]:
            if args.counts:
                print("{:11}|{:10} |{:10} |{:10} |{:10}".format(
                    metric,
                    evaluation[metric].correct,
                    evaluation[metric].gold_total,
                    evaluation[metric].system_total,
                    evaluation[metric].aligned_total or (evaluation[metric].correct if metric == "Words" else "")
                ))
            else:
                print("{:11}|{:10.2f} |{:10.2f} |{:10.2f} |{}".format(
                    metric,
                    100 * evaluation[metric].precision,
                    100 * evaluation[metric].recall,
                    100 * evaluation[metric].f1,
                    "{:10.2f}".format(100 * evaluation[metric].aligned_accuracy) if evaluation[
                                                                                        metric].aligned_accuracy is not None else ""
                ))


if __name__ == "__main__":
    main()
