import itertools
import speechbrain as sb
from speechbrain.dataio.encoder import CategoricalEncoder
from speechbrain.tokenizers.SentencePiece import SentencePiece
import torchaudio
import torch


def generate_seq_labels(poss, deps, max_span) -> str:
    '''
    Return special symbols representing pos, dep, and head_position

    >>> generate_seq_labels(4, 3, 2)
    '<s>,</s>,<POS0>,<POS1>,<POS2>,<POS3>,<DEP0>,<DEP1>,<DEP2>,<L0>,<R0>,<L1>,<R1>'
    '''
    pos_symbols = [f'<POS{i}>' for i in range(poss)]
    dep_symbols = [f'<DEP{i}>' for i in range(deps)]
    # head_position_symbols = list(itertools.chain.from_iterable(
    #     [[f'<L{i}>', f'<R{i}>', f'</L{i}>', f'</R{i}>'] for i in range(max_span)]
    # ))
    head_position_symbols = list(itertools.chain.from_iterable(
        [[f'<L{i}>', f'<R{i}>'] for i in range(max_span)]
    ))
    return ",".join(["<s>", "</s>"] + pos_symbols + dep_symbols + head_position_symbols)


def generate_target_seq_word(wrd, pos, gov, dep):
    '''
    Return the tokenized target sequence.
    The relative position of the head is represented in a word level.

    >>> generate_target_seq_word("A VOULU PARTAGER", "0 1 2", "2 0 2", "2 3 4")
    '<s> A<POS0><R1><DEP2> VOULU<POS1><L2><DEP3> PARTAGER<POS2><L1><DEP4></s>'
    >>> generate_target_seq_word("EST UN PROBLÈME", "4 3 2", "0 3 1", "1 2 3")
    '<s> EST<POS4><L1><DEP1> UN<POS3><R1><DEP2> PROBLÈME<POS2><L2><DEP3></s>'
    '''

    def encode_head_position(h):
        encoded = ""
        if h <= 0:
            encoded += f'<L{-h}>'
        else:
            encoded += f'<R{h}>'
        return encoded

    target_seq = "<s>"

    for i, (w, pi, g, di) in enumerate(zip(wrd.split(' '), pos.split(' '), gov.split(' '), dep.split(' '))):
        target_seq += f" {w}<POS{pi}>"
        head_position = (int(g) - 1) - int(i)
        target_seq += encode_head_position(head_position)
        target_seq += f"<DEP{di}>"
    target_seq += "</s>"
    return target_seq


# Define custom data procedure
def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    # 1. Define datasets
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"]
    )

    # we sort training data to speed up training and get better results.
    train_data = train_data.filtered_sorted(
        sort_key="duration",
        key_max_value={"duration": hparams["avoid_if_longer_than"]},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"]
    )
    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_csv"]
    )

    # We also sort the validation/test data so it is faster to validate
    valid_data = valid_data.filtered_sorted(sort_key="duration")
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        '''
        Audio Pipeline
        Parameters
        ----------
        wav : the wav file path

        Returns
        resampled : return the raw signal from the file with the right sampling ( 16Khz)
        -------

        '''
        info = torchaudio.info(wav)
        sig = sb.dataio.dataio.read_audio(wav)
        resampled = torchaudio.transforms.Resample(
            info.sample_rate, hparams["sample_rate"],
        )(sig)
        return resampled

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define target sequence pipeline:

    # Define label encoder
    pos_set = set()
    dep_set = set()
    with train_data.output_keys_as(["pos", "dep"]):
        for d in train_data:
            for pos in d["pos"].split(' '):
                pos_set.add(pos)
            for dep in d["dep"].split(' '):
                dep_set.add(dep)

    pos_encoder = CategoricalEncoder()
    pos_encoder.update_from_iterable(sorted(pos_set))
    pos_encoder.add_unk()
    dep_encoder = CategoricalEncoder()
    dep_encoder.update_from_iterable(sorted(dep_set))
    dep_encoder.add_unk()

    # Define tokenizer
    # max_frames = int(hparams["wav2vec2_freq"]) * int(hparams["avoid_if_longer_than"])
    with train_data.output_keys_as(["wrd"]):
        max_span = 0
        for d in train_data:
            max_span = max(max_span, len(d["wrd"].split(' ')))
    user_defined_symbols = generate_seq_labels(len(pos_encoder), len(dep_encoder), max_span)
    tokenizer = SentencePiece(
        model_dir=hparams["save_folder"],
        vocab_size=hparams["output_neurons"],
        annotation_train=hparams["train_csv"],
        annotation_read="wrd",
        model_type=hparams["token_type"],
        character_coverage=hparams["character_coverage"],
        user_defined_symbols=user_defined_symbols,
        bos_id=hparams["bos_index"],
        eos_id=hparams["eos_index"],
        add_dummy_prefix=False
    )

    @sb.utils.data_pipeline.takes("wrd", "pos", "gov", "dep")
    @sb.utils.data_pipeline.provides("target_seqs", "target_seqs_encoded", "tokens")
    def target_seq_pipeline(wrd, pos, gov, dep):
        '''
        Dependency Parsing by Labeled Sequence Prediction
        Parameters
        ----------
        wrd : the word contained in the CSV file
        pos : the part of speech in the CSV file
        gov : the gov/head label in the CSV file
        dep : the syntactic function in the CSV file

        Returns
        target_seq : the tokenized labeled sequence formatted as follows:
        <s> w_1<pos_1><h_1><rel_1> w2<pos_2><h_2><rel_2> ...</s>
        -------
        '''

        pos_encoded = ' '.join([str(pos_encoder.encode_label(p)) for p in pos.split(' ')])
        dep_encoded = ' '.join([str(dep_encoder.encode_label(d)) for d in dep.split(' ')])

        target_seqs = generate_target_seq_word(wrd, pos_encoded, gov, dep_encoded)
        yield target_seqs
        target_seqs_encoded = torch.LongTensor(tokenizer.sp.encode_as_ids(target_seqs))
        yield target_seqs_encoded
        tokens_list = tokenizer.sp.encode_as_ids(wrd)
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, target_seq_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "duration", "sig", "target_seqs", "target_seqs_encoded", "tokens", "wrd", "pos", "gov", "dep"]
    )

    return train_data, valid_data, test_data, tokenizer, pos_encoder, dep_encoder
