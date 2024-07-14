import os
import sys
import dataset
import evaluate
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main, if_main_process
import torch
import wandb
import subprocess


def get_id_from_CoNLLfile(path):
    '''
    Get the sentence id from the conll file in the order
    Will be used to write in the same order for comparaison sakes.
    '''
    sent_id = []
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            if line.startswith("# sent_id"):
                field = line.split("=")
                sent_id.append(field[1].replace(" ", "").replace("\n", ""))
    return sent_id


class SpeechParser(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)

        # Forward pass

        # HuggingFace pretrained model
        feats = self.modules.wav2vec2(wavs, wav_lens)

        x = self.modules.enc(feats)

        # Compute outputs
        sequences = None
        logits = self.modules.ctc_lin(x)

        p_ctc = self.hparams.log_softmax(logits)
        if stage == sb.Stage.VALID or stage == sb.Stage.TEST:
            sequences = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
        return p_ctc, wav_lens, sequences

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        def _get_special_token_final_id(tokenizer):
            for id in range(tokenizer.sp.get_piece_size()):
                piece = tokenizer.sp.id_to_piece(id)
                if not piece.startswith("<"):
                    return id - 1
            raise ValueError("spm vocabulary is composed of special tokens only. Check vocab file.")

        p_ctc, wav_lens, sequences = predictions

        ids = batch.id
        seq, seq_lens = batch.target_seqs_encoded

        loss_ctc = self.hparams.ctc_cost(p_ctc, seq, wav_lens, seq_lens)
        loss = loss_ctc

        if stage != sb.Stage.TRAIN:
            # print(ids)

            # Decode predicted sequences
            # After sufficient training, items in sequences look like `<s> I<POS0><R1><DEP0> go<POS1><L2><DEP1></s>`
            # (the only possible places where blanks appear are "just before words")
            predicted_seqs = [
                self.tokenizer.sp.decode_ids(seq) for seq in sequences
            ]

            # Decode words
            # by filtering out special tokens (eg: <s>, <POS0>, <L10>) from predicted sequences
            special_token_final_id = _get_special_token_final_id(tokenizer)
            predicted_words = [
                self.tokenizer.sp.decode_ids(list(filter(lambda x: x > special_token_final_id, seq))).split(" ") for seq in sequences
            ]

            target_words = [wrd.split(" ") for wrd in batch.wrd]

            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)
            self.evaluator.decode(ids, predicted_seqs)

        return loss

    def fit_batch(self, batch):
        should_step = self.step % self.grad_accumulation_factor == 0

        # Managing automatic mixed precision
        if self.auto_mix_prec:
            self.wav2vec_optimizer.zero_grad()
            self.model_optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                with self.no_sync():
                    outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            with self.no_sync(not should_step):
                self.scaler.scale(
                    loss / self.grad_accumulation_factor
                ).backward()
            if should_step:
                if not self.hparams.freeze_wav2vec:
                    self.scaler.unscale_(self.wav2vec_optimizer)
                self.scaler.unscale_(self.model_optimizer)
                if self.check_gradients(loss):
                    self.scaler.step(self.wav2vec_optimizer)
                    self.scaler.step(self.model_optimizer)
                self.scaler.update()
                self.optimizer_step += 1
        else:
            with self.no_sync():
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            (loss / self.grad_accumulation_factor).backward()
            if should_step:
                if self.check_gradients(loss):
                    self.wav2vec_optimizer.step()
                    self.model_optimizer.step()
                self.wav2vec_optimizer.zero_grad()
                self.model_optimizer.zero_grad()
                self.optimizer_step += 1

        wandb.log({"loss": loss})

        return loss.detach().cpu()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")
            if stage == sb.Stage.VALID:
                predicted_conllu_path = self.hparams.dev_output_conllu
                gold_conllu_path = self.hparams.dev_gold_conllu
            else:
                predicted_conllu_path = self.hparams.test_output_conllu
                gold_conllu_path = self.hparams.test_gold_conllu
            gold_trans = gold_conllu_path + "_trans"
            order = get_id_from_CoNLLfile(gold_conllu_path)
            print(f"writing dev file in {predicted_conllu_path}")
            self.evaluator.write_to_file(predicted_conllu_path, order)
            predicted_trans = predicted_conllu_path + "_trans"
            self.evaluator.write_trans_to_file(predicted_trans, order)
            # create file named predicted_trans.sgml
            sclite_command = ["sclite", "-F", "-i", "wsj", "-r", gold_trans,
                              "-h", predicted_trans, "-o", "sgml"]
            subprocess.run(sclite_command,
                           cwd=os.path.dirname(os.path.abspath(__file__)), check=True)
            if stage == sb.Stage.VALID:
                metrics_dict, _, _ = self.evaluator.evaluate_conllu(
                    gold_conllu_path,
                    predicted_conllu_path,
                    f"{predicted_trans}.sgml"
                )
            elif stage == sb.Stage.TEST:
                metrics_dict, pos_stat, uas_list = self.evaluator.evaluate_conllu(
                    gold_conllu_path,
                    predicted_conllu_path,
                    f"{predicted_trans}.sgml",
                    analysis=True
                )
            stage_stats["LAS"] = metrics_dict["LAS"].f1 * 100
            stage_stats["UAS"] = metrics_dict["UAS"].f1 * 100
            stage_stats["SER"] = metrics_dict["seg_error_rate"].precision * 100
            stage_stats["SENTENCES"] = metrics_dict["Sentences"].precision * 100
            stage_stats["Tokens"] = metrics_dict["Tokens"].precision * 100
            stage_stats["UPOS"] = metrics_dict["UPOS"].f1 * 100

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr_model, new_lr_model = self.hparams.lr_annealing_model(
                stage_stats["loss"]
            )
            old_lr_wav2vec, new_lr_wav2vec = self.hparams.lr_annealing_wav2vec(
                stage_stats["loss"]
            )
            sb.nnet.schedulers.update_learning_rate(
                self.model_optimizer, new_lr_model
            )
            sb.nnet.schedulers.update_learning_rate(
                self.wav2vec_optimizer, new_lr_wav2vec
            )
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr_model": old_lr_model,
                    "lr_wav2vec": old_lr_wav2vec,
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            wandb_stats = {"epoch": epoch}
            wandb_stats = {**wandb_stats, **stage_stats}  # fuse dict
            wandb.log(wandb_stats)
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                with open(self.hparams.wer_file, "w") as w:
                    self.wer_metric.write_stats(w)
                # save pos_stat to file
                analysis_dir = f"{self.hparams.output_folder}/analysis"
                os.makedirs(analysis_dir, exist_ok=True)
                for pos, stat in pos_stat.items():
                    filename = f"{pos}_HEAD"
                    with open(f"{analysis_dir}/{filename}", "w") as f:
                        f.write(f"HEAD\ttotal\tcorrect\n")
                        for headpos, num in stat['HEAD'].items():
                            f.write(f"{headpos}\t{num['gold']}\t{num['pred']}\n")
                    filename = f"{pos}_DEPREL"
                    with open(f"{analysis_dir}/{filename}", "w") as f:
                        f.write(f"DEPREL\ttotal\tcorrect\n")
                        for deprel, num in stat['DEPREL'].items():
                            f.write(f"{deprel}\t{num['gold']}\t{num['pred']}\n")
                # save uas to file
                with open(f"{analysis_dir}/uas.txt", mode='w') as f:
                    for uas in uas_list:
                        f.write(str(uas) + "\n")

    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"
        self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
            self.modules.wav2vec2.parameters()
        )
        self.model_optimizer = self.hparams.model_opt_class(
            self.hparams.model.parameters()
        )

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "wav2vec_opt", self.wav2vec_optimizer
            )
            self.checkpointer.add_recoverable("modelopt", self.model_optimizer)

    def zero_grad(self, set_to_none=False):
        self.wav2vec_optimizer.zero_grad(set_to_none)
        self.model_optimizer.zero_grad(set_to_none)


if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Logging with wandb
    wandb.init(project="textless", group=hparams['dataset'], name=str(hparams['seed']))

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Create the datasets objects as well as tokenization and encoding :-D
    train_data, valid_data, test_data, tokenizer, pos_encoder, dep_encoder = dataset.dataio_prepare(hparams)

    # Trainer initialization
    asr_brain = SpeechParser(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    evaluator = evaluate.TreeEvaluator(pos_encoder, dep_encoder)

    # Adding objects to trainer.
    asr_brain.tokenizer = tokenizer
    asr_brain.evaluator = evaluator

    # Training
    try:
        asr_brain.fit(
            asr_brain.hparams.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=hparams["dataloader_options"],
            valid_loader_kwargs=hparams["test_dataloader_options"],
        )
    except RuntimeError as e:  # Memory Leak
        import gc

        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    pass
                    # print(type(obj), obj.size())
            except:
                pass
        raise RuntimeError() from e

    # Test

    asr_brain.hparams.wer_file = hparams["output_folder"] + "/wer_test.txt"
    asr_brain.evaluate(
        test_data,
        min_key="WER",
        test_loader_kwargs=hparams["test_dataloader_options"],
    )

    # transcribe

    # asr_brain.transcribe_dataset(dataset=test_data,  # Must be obtained from the dataio_function
    #                              min_key="WER",
    #                              loader_kwargs=hparams["test_dataloader_options"], )
