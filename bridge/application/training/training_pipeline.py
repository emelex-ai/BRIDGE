import gc
import json
import logging
import os
import random
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import torch
from tqdm import tqdm

from bridge.application.training.ortho_metrics import calculate_orth_metrics
from bridge.application.training.phon_metrics import calculate_phon_metrics
from bridge.core.phonreps import load_phoneme_table
from bridge.domain.data import BridgeDataset
from bridge.domain.datamodels import EncodingComponent, TrainingConfig, TrainingEvent
from bridge.domain.model import Model
from bridge.infra.metrics.metrics_logger import MetricsLogger
from bridge.utils import device_manager

# Per-step results: loss tensors + scalar metrics + the JSON-encoded `word`.
type MetricsDict = dict[str, torch.Tensor | float | str]
# Per-epoch results: tensors and floats only. `word` is dropped during
# accumulation, and timing values are added as floats.
type NumericMetrics = dict[str, torch.Tensor | float]

min_interval = 1


class TrainingPipeline:
    def __init__(
        self,
        model: Model,
        training_config: TrainingConfig,
        dataset: BridgeDataset,
        metrics_logger: MetricsLogger,
    ):
        self.logger = logging.getLogger(__name__)
        self.metrics_logger = metrics_logger
        self.training_config = training_config
        self.dataset = dataset
        self.test_dataset = None
        if self.training_config.test_data_path:
            test_dataset_config = self.dataset.dataset_config.model_copy()
            test_dataset_config.dataset_filepath = self.training_config.test_data_path
            self.test_dataset = BridgeDataset(
                dataset_config=test_dataset_config,
                gcs_client=self.dataset.gcs_client,
                # Reuse the train tokenizer: building a second one re-parses the
                # pronunciation lexicons (~1 s, ~81 MB), and the duplicate also inflated
                # the periodic gc.collect() below from ~109 ms to ~188 ms.
                tokenizer=self.dataset.tokenizer,
            )
        self.device = device_manager.device
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
        self.train_slices, self.val_slices = self.create_data_slices()
        self.phon_reps = load_phoneme_table(device=self.device).phonetic_features

        self.start_epoch = 0
        if training_config.checkpoint_path:
            self.load_model(training_config.checkpoint_path)

    def create_data_slices(self):
        # Kept on the instance: `_shuffle_training_partition` reorders exactly the indices
        # below this point, so the two must not compute it separately and drift.
        self.cutpoint = cutpoint = int(len(self.dataset) * self.training_config.train_test_split)
        train_slices = [
            slice(i, min(i + self.training_config.batch_size_train, cutpoint))
            for i in range(0, cutpoint, self.training_config.batch_size_train)
        ]
        val_slices = [
            slice(i, min(i + self.training_config.batch_size_val, len(self.dataset)))
            for i in range(cutpoint, len(self.dataset), self.training_config.batch_size_val)
        ]
        return train_slices, val_slices

    def forward(
        self, orthography: EncodingComponent, phonology: EncodingComponent
    ) -> dict[str, torch.Tensor]:
        if self.training_config.training_pathway == "o2p":
            return self.model(
                task="o2p",
                orth_enc_input=orthography.enc_input_ids,
                orth_enc_pad_mask=orthography.enc_pad_mask,
                phon_dec_input=phonology.dec_input_ids,
                phon_dec_pad_mask=phonology.dec_pad_mask,
            )
        elif self.training_config.training_pathway == "op2op":
            return self.model(
                task="op2op",
                orth_enc_input=orthography.enc_input_ids,
                orth_enc_pad_mask=orthography.enc_pad_mask,
                orth_dec_input=orthography.dec_input_ids,
                orth_dec_pad_mask=orthography.dec_pad_mask,
                phon_enc_input=phonology.enc_input_ids,
                phon_enc_pad_mask=phonology.enc_pad_mask,
                phon_dec_input=phonology.dec_input_ids,
                phon_dec_pad_mask=phonology.dec_pad_mask,
            )
        elif self.training_config.training_pathway == "p2o":
            return self.model(
                task="p2o",
                phon_enc_input=phonology.enc_input_ids,
                phon_enc_pad_mask=phonology.enc_pad_mask,
                orth_dec_input=orthography.dec_input_ids,
                orth_dec_pad_mask=orthography.dec_pad_mask,
            )
        elif self.training_config.training_pathway == "p2p":
            return self.model(
                task="p2p",
                phon_enc_input=phonology.enc_input_ids,
                phon_enc_pad_mask=phonology.enc_pad_mask,
                phon_dec_input=phonology.dec_input_ids,
                phon_dec_pad_mask=phonology.dec_pad_mask,
            )
        else:
            raise ValueError(f"Unknown training_pathway: {self.training_config.training_pathway!r}")

    def compute_loss(
        self,
        logits: dict[str, torch.Tensor],
        orthography: EncodingComponent,
        phonology: EncodingComponent,
    ) -> dict[str, torch.Tensor]:
        orth_loss: torch.Tensor | None = None
        phon_loss: torch.Tensor | None = None

        vocab = self.model.model_config.vocab

        # Calculate phon_loss if applicable
        if self.training_config.training_pathway in ["o2p", "op2op", "p2p"]:
            phon_loss = torch.nn.CrossEntropyLoss(ignore_index=vocab.phon_pad_id)(
                logits["phon"], phonology.phon_targets
            )

        # Calculate orth_loss if applicable
        if self.training_config.training_pathway in ["p2o", "op2op"]:
            # Teacher forcing: decoder position i is scored against the token at i+1. The
            # character tokenizer lays each sequence out as
            #     enc = [LANG, BOS, ...chars, EOS, PAD...]
            #     dec = [LANG, BOS, ...chars,      PAD...]
            # so the encoder ids shifted left by one give the next token for every decoder
            # position, at the same width. The `[BOS] -> first character` pair this creates
            # is what generation needs: `orthography_decoder_loop` seeds a lone [BOS] and
            # the token it samples next must be the word's first character. See
            # docs/decisions/0004-orthographic-teacher-forcing-alignment.md.
            orth_target = orthography.enc_input_ids[:, 1:]
            self._check_orth_target_width(logits["orth"], orth_target)
            orth_loss = torch.nn.CrossEntropyLoss(ignore_index=vocab.orth_pad_id)(
                logits["orth"], orth_target
            )

        if orth_loss is not None and phon_loss is not None:
            total_loss = orth_loss + phon_loss
        elif orth_loss is not None:
            total_loss = orth_loss
        elif phon_loss is not None:
            total_loss = phon_loss
        else:
            raise ValueError(
                f"No loss configured for training_pathway={self.training_config.training_pathway!r}"
            )

        loss_dict: dict[str, torch.Tensor] = {"loss": total_loss}
        if orth_loss is not None:
            loss_dict["orth_loss"] = orth_loss
        if phon_loss is not None:
            loss_dict["phon_loss"] = phon_loss

        return loss_dict

    @staticmethod
    def _check_orth_target_width(orth_logits: torch.Tensor, orth_target: torch.Tensor) -> None:
        """Fail with both shapes named when the target and the logits disagree.

        Left to ``CrossEntropyLoss`` this reads ``Expected target size [8, 8], got [8, 7]``,
        which names neither tensor nor where either came from. That message is what issue
        #225 presented as, and it cost more to diagnose than it should have.
        """
        if orth_target.shape[1] != orth_logits.shape[-1]:
            raise ValueError(
                "Orthographic loss target and logits disagree on how many positions the "
                f"decoder scored: target {tuple(orth_target.shape)} covers "
                f"{orth_target.shape[1]} positions, logits {tuple(orth_logits.shape)} cover "
                f"{orth_logits.shape[-1]}. The target must be enc_input_ids shifted left by "
                "one, giving exactly one target per decoder input position."
            )

    def compute_metrics(
        self,
        logits: dict[str, torch.Tensor],
        orthography: EncodingComponent,
        phonology: EncodingComponent,
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if self.training_config.training_pathway in ["o2p", "op2op", "p2p"]:
            metrics.update(
                calculate_phon_metrics(
                    logits,
                    phonology,
                    self.phon_reps,
                    phon_pad_id=self.model.model_config.vocab.phon_pad_id,
                )
            )

        if self.training_config.training_pathway in ["op2op", "p2o"]:
            metrics.update(
                calculate_orth_metrics(
                    logits, orthography, orth_pad_id=self.model.model_config.vocab.orth_pad_id
                )
            )

        return metrics

    def single_step(
        self,
        dataset: BridgeDataset,
        batch_slice: slice,
        calculate_metrics: bool = False,
    ) -> MetricsDict:
        num_chunks = self.training_config.num_chunks if self.training_config.num_chunks else 1
        # Fast path when not using accumulated gradients
        if num_chunks == 1:
            # Zero gradients
            if self.model.training:
                self.optimizer.zero_grad()

            # Process the entire batch at once
            batch = dataset[batch_slice]
            orthography, phonology = batch.orthographic, batch.phonological

            # Forward pass
            logits = self.forward(orthography, phonology)

            # Compute loss (no scaling needed). Kept as `dict[str, Tensor]` so
            # `.backward()` is well-typed; widened to MetricsDict only after.
            loss_metrics = self.compute_loss(logits, orthography, phonology)

            # Backward pass
            if self.model.training:
                loss_metrics["loss"].backward()
                self.optimizer.step()
                self.optimizer.zero_grad()  # Reset gradients after update

            metrics: MetricsDict = dict(loss_metrics)

            # Calculate additional metrics if needed
            if calculate_metrics:
                metrics.update(self.compute_metrics(logits, orthography, phonology))

            if self.metrics_logger.metrics_config.batch_metrics:
                self.metrics_logger.log_metrics(metrics, "BATCH")

            metrics["word"] = json.dumps(dataset.words[batch_slice])
            return metrics

        # Original accumulated gradients path for num_chunks > 1
        accumulated_losses: dict[str, torch.Tensor] = {}
        sub_slices = self._create_sub_slices(batch_slice, num_chunks=num_chunks)

        # Zero gradients once at the beginning
        self.optimizer.zero_grad()

        # Losses accumulate across every sub-batch, but metrics are computed once, from
        # the final sub-batch, so the loop carries that one forward. Distinct names from
        # the `num_chunks == 1` path above, which binds `logits`/`orthography`/`phonology`
        # in this same function scope.
        last_logits: dict[str, torch.Tensor] | None = None
        last_orthography: EncodingComponent | None = None
        last_phonology: EncodingComponent | None = None

        for sub_slice in sub_slices:
            batch = dataset[sub_slice]
            last_orthography, last_phonology = batch.orthographic, batch.phonological

            # Forward pass
            last_logits = self.forward(last_orthography, last_phonology)

            # Compute loss with scaled factor
            sub_metrics = self.compute_loss(last_logits, last_orthography, last_phonology)
            loss = sub_metrics["loss"] / num_chunks  # Scale loss by number of chunks

            # Backward pass (accumulate gradients)
            if self.model.training:
                loss.backward()

            # Update metrics
            if not accumulated_losses:
                accumulated_losses = dict(sub_metrics)
            else:
                for k, v in sub_metrics.items():
                    accumulated_losses[k] = accumulated_losses[k] + v

        # Only step optimizer after processing all sub-batches
        if self.model.training:
            self.optimizer.step()
            self.optimizer.zero_grad()

        # `_create_sub_slices` yields nothing for an empty batch slice, leaving the
        # loop-carried values unset. Raise rather than assert: `assert` is stripped under
        # `python -O`, which would let None reach `compute_metrics` as an AttributeError.
        if last_logits is None or last_orthography is None or last_phonology is None:
            raise ValueError(
                f"No sub-batches produced for batch_slice {batch_slice} with "
                f"num_chunks={num_chunks}; the slice is empty."
            )

        accumulated_metrics: MetricsDict = dict(accumulated_losses)
        if calculate_metrics:
            accumulated_metrics.update(
                self.compute_metrics(last_logits, last_orthography, last_phonology)
            )

        if self.metrics_logger.metrics_config.batch_metrics:
            self.metrics_logger.log_metrics(accumulated_metrics, "BATCH")

        accumulated_metrics["word"] = json.dumps(dataset.words[batch_slice])
        return accumulated_metrics

    def _create_sub_slices(self, batch_slice: slice, num_chunks: int) -> list[slice]:
        """Split a slice into smaller slices."""
        start, stop = batch_slice.start, batch_slice.stop
        size = stop - start
        chunk_size = max(1, size // num_chunks)

        sub_slices = []
        for i in range(0, size, chunk_size):
            sub_start = start + i
            sub_stop = min(start + i + chunk_size, stop)
            sub_slices.append(slice(sub_start, sub_stop))

        return sub_slices

    def train_steps(self, epoch: int) -> Iterator[TrainingEvent]:
        """Run one training epoch, yielding after every optimizer step.

        Public. This is the seam a caller reaches for to own the loop: checkpoint on a step
        count, stop early on a loss, log at whatever cadence suits. The pipeline decides
        none of that. :meth:`run_train_val_loop` is a thin wrapper over this.

        Shuffling is deliberately *not* done here. It belongs to the epoch, and a caller
        driving `train_steps` directly across several epochs would otherwise get a
        reordering it did not ask for. `run_train_val_loop` calls
        `_shuffle_training_partition` before each epoch; a caller doing their own loop calls
        it themselves, or does not.
        """
        self.model.train()
        last_update_time = time.time()
        progress_bar = tqdm(
            self.train_slices,
            desc=f"Training Epoch {epoch + 1}",
            mininterval=min_interval,
        )
        for step, batch_slice in enumerate(progress_bar):
            # Run garbage collection to free up memory
            if step % 10 == 0:
                gc.collect()

            metrics = self.single_step(
                self.dataset,
                batch_slice,
                self.metrics_logger.metrics_config.training_metrics,
            )
            current_time = time.time()
            if current_time - last_update_time > min_interval:
                progress_bar.set_postfix(
                    {
                        key: f"{value:.4f}"
                        for key, value in metrics.items()
                        if not isinstance(value, str)
                    }
                )
                last_update_time = current_time
            yield TrainingEvent(phase="train", epoch=epoch, step=step, metrics=metrics)

    def train_single_epoch(self, epoch: int) -> NumericMetrics:
        """Run one training epoch and return its mean metrics, `train_` prefixed.

        Kept as the aggregating convenience over :meth:`train_steps`, for callers who want
        an epoch number rather than a stream.
        """
        start = time.time()
        total_metrics: NumericMetrics = {}
        for event in self.train_steps(epoch):
            step_numeric: NumericMetrics = {
                key: value for key, value in event.metrics.items() if not isinstance(value, str)
            }
            if not total_metrics:
                total_metrics = step_numeric
            else:
                for key, value in step_numeric.items():
                    total_metrics[key] = total_metrics[key] + value
        for key in total_metrics:
            total_metrics[key] = total_metrics[key] / len(self.train_slices)
        total_metrics["time_per_step"] = (time.time() - start) / len(self.train_slices)
        total_metrics["time_per_epoch"] = (time.time() - start) * len(self.train_slices)
        return {"train_" + str(key): val for key, val in total_metrics.items()}

    def validate_single_epoch(self, epoch: int) -> NumericMetrics:
        self.model.eval()
        start = time.time()
        last_update_time = time.time()
        progress_bar = tqdm(
            self.val_slices,
            desc=f"Validating Epoch {epoch + 1}",
            mininterval=min_interval,
        )

        with torch.no_grad():
            total_metrics: NumericMetrics = {}
            for _step, batch_slice in enumerate(progress_bar):
                metrics = self.single_step(
                    self.dataset,
                    batch_slice,
                    self.metrics_logger.metrics_config.validation_metrics,
                )
                step_numeric: NumericMetrics = {
                    key: value for key, value in metrics.items() if not isinstance(value, str)
                }
                current_time = time.time()
                if current_time - last_update_time > min_interval:
                    progress_bar.set_postfix(
                        {key: f"{value:.4f}" for key, value in step_numeric.items()}
                    )
                    last_update_time = current_time
                if not total_metrics:
                    total_metrics = step_numeric
                else:
                    for key, value in step_numeric.items():
                        total_metrics[key] = total_metrics[key] + value
            for key in total_metrics:
                total_metrics[key] = total_metrics[key] / len(self.val_slices)
        total_metrics["time_per_step"] = (time.time() - start) / len(self.val_slices)
        total_metrics["time_per_epoch"] = (time.time() - start) * len(self.val_slices)
        return {"valid_" + str(key): val for key, val in total_metrics.items()}

    def test_single_epoch(self, epoch: int) -> NumericMetrics:
        self.model.eval()
        start = time.time()
        last_update_time = time.time()
        if self.test_dataset is None:
            raise ValueError("Test dataset not provided in the configuration.")

        # Create test slices based on batch size
        test_slices = [
            slice(
                i,
                min(i + self.training_config.batch_size_train, len(self.test_dataset)),
            )
            for i in range(0, len(self.test_dataset), self.training_config.batch_size_train)
        ]
        progress_bar = tqdm(
            test_slices, desc=f"Testing Epoch {epoch + 1}", mininterval=min_interval
        )

        with torch.no_grad():
            total_metrics: NumericMetrics = {}
            for _step, batch_slice in enumerate(progress_bar):
                metrics = self.single_step(
                    self.test_dataset,
                    batch_slice,
                    self.metrics_logger.metrics_config.validation_metrics,
                )
                step_numeric: NumericMetrics = {
                    key: value for key, value in metrics.items() if not isinstance(value, str)
                }
                current_time = time.time()
                if current_time - last_update_time > min_interval:
                    progress_bar.set_postfix(
                        {key: f"{value:.4f}" for key, value in step_numeric.items()}
                    )
                    last_update_time = current_time
                if not total_metrics:
                    total_metrics = step_numeric
                else:
                    for key, value in step_numeric.items():
                        total_metrics[key] = total_metrics[key] + value
            for key in total_metrics:
                total_metrics[key] = total_metrics[key] / len(test_slices)
        total_metrics["time_per_step"] = (time.time() - start) / len(test_slices)
        total_metrics["time_per_epoch"] = (time.time() - start) * len(test_slices)
        return {"test_" + str(key): val for key, val in total_metrics.items()}

    def run_train_val_loop(self, num_epochs: int | None = None) -> Iterator[TrainingEvent]:
        """Train, validating each epoch, yielding a record at every step and boundary.

        The convenience loop, and a thin one: everything it does is available separately as
        :meth:`train_steps`, :meth:`validate_single_epoch`, :meth:`test_single_epoch` and
        :meth:`_shuffle_training_partition`. A caller who wants a different loop writes it
        out of those rather than passing flags into this one.

        It writes no checkpoints. When to save, where, and under what name are the caller's
        to decide, from the stream:

            for event in pipeline.run_train_val_loop(num_epochs=3):
                if event.phase == "train" and event.step % 100 == 0:
                    pipeline.save_checkpoint(runs / f"step_{event.step}.pth", event.epoch)
                elif event.phase == "epoch":
                    pipeline.save_checkpoint(runs / f"epoch_{event.epoch}.pth", event.epoch)

        Four phases arrive, and the ``epoch`` one is always last for its epoch. Its metrics
        are the merged aggregate this generator used to yield before it yielded per step, so
        a caller who only wants epoch rows filters on that phase and is otherwise unchanged.

        Args:
            num_epochs: Overrides `training_config.num_epochs` for this call. Counting still
                starts at `self.start_epoch`, so a resumed run does the remaining epochs.
        """
        total = self.training_config.num_epochs if num_epochs is None else num_epochs
        for epoch in range(self.start_epoch, total):
            self._shuffle_training_partition(epoch)

            epoch_metrics: NumericMetrics = {}
            start = time.time()
            step_total: NumericMetrics = {}
            steps = 0
            for event in self.train_steps(epoch):
                steps += 1
                numeric: NumericMetrics = {
                    key: value for key, value in event.metrics.items() if not isinstance(value, str)
                }
                if not step_total:
                    step_total = numeric
                else:
                    for key, value in numeric.items():
                        step_total[key] = step_total[key] + value
                yield event
            if steps:
                for key in step_total:
                    step_total[key] = step_total[key] / steps
                step_total["time_per_step"] = (time.time() - start) / steps
                step_total["time_per_epoch"] = time.time() - start
                epoch_metrics.update({"train_" + str(k): v for k, v in step_total.items()})

            if self.val_slices:
                metrics = self.validate_single_epoch(epoch)
                epoch_metrics.update(metrics)
                yield TrainingEvent(phase="validation", epoch=epoch, metrics=dict(metrics))
            if self.test_dataset:
                metrics = self.test_single_epoch(epoch)
                epoch_metrics.update(metrics)
                yield TrainingEvent(phase="test", epoch=epoch, metrics=dict(metrics))

            self.metrics_logger.log_metrics(epoch_metrics, "EPOCH")
            yield TrainingEvent(phase="epoch", epoch=epoch, metrics=dict(epoch_metrics))

        self.metrics_logger.save()

    def _shuffle_training_partition(self, epoch: int) -> None:
        """Reorder the training words in place, leaving the validation tail untouched.

        Without this every epoch iterates the data in file order, identically, for every
        model trained on it. For an alphabetically sorted lexicon that means training on all
        the "a" words first, every epoch. It also means a seed intended to vary data order
        contributes exactly nothing, so an experiment treating order as an independent
        variable is silently measuring a constant.

        The validation tail stays where it is, so validation scores remain comparable across
        epochs. `create_data_slices` computed its slices as index ranges once, in `__init__`,
        and reordering in place keeps them valid. `BridgeDataset`'s encoding cache is keyed
        by (word, language) rather than by index, so it survives the reordering too.

        `BridgeDataset.shuffle` draws from the global `random` module, so the per-epoch seed
        has to go through `random.seed`. Deriving it from the config seed and the epoch gives
        a different order each epoch while keeping the whole run reproducible.
        """
        if not self.training_config.shuffle_each_epoch:
            return
        if self.training_config.seed is not None:
            random.seed(self.training_config.seed * 10_000 + epoch)
        self.dataset.shuffle(self.cutpoint)

    def save_checkpoint(self, path: str | Path, epoch: int) -> Path:
        """Write the full checkpoint bundle to ``path``, and return where it went.

        No policy. The caller chooses when to save and what to call the file; the pipeline
        knows what belongs in one. That split is why this takes a path rather than a run
        name and a ``save_every`` cadence: two runs sharing an artifacts directory used to
        overwrite each other silently, because the filename depended only on the epoch, and
        no naming scheme the library picks can be right for every experiment.

        A relative path resolves against `training_config.model_artifacts_dir`, so the
        common case stays short. Parent directories are created here, which is the point of
        first write now that validating a config no longer touches the filesystem.

        `epoch` is recorded in the bundle and is what `load_model` resumes from, so it must
        be the epoch these weights finished, not the one about to start.
        """
        destination = Path(path)
        if not destination.is_absolute():
            destination = Path(self.training_config.model_artifacts_dir) / destination
        destination.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_config": self.model.model_config,
                "dataset_config": self.dataset.dataset_config,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": epoch,
            },
            destination,
        )
        if self.dataset.gcs_client:
            self.dataset.gcs_client.upload_file(
                os.environ["BUCKET_NAME"],
                str(destination),
                f"{self.training_config.gcs_path}/models/{destination.name}",
            )
        return destination

    def load_model(self, model_path: str):
        try:
            import bridge
            import bridge.domain as bridge_domain
            import bridge.domain.datamodels as bridge_datamodels
            import bridge.domain.datamodels.model_config as old_module_reference

            sys.modules["src"] = bridge
            sys.modules["src.domain"] = bridge_domain
            sys.modules["src.domain.datamodels"] = bridge_datamodels
            sys.modules["src.domain.datamodels.model_config"] = old_module_reference

            checkpoint = torch.load(model_path, weights_only=False)
            self._warn_on_phoneme_table_drift(checkpoint, model_path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            self._set_start_epoch(checkpoint, model_path)

            return True
        except Exception as e:
            self.logger.error(f"Error loading checkpoint {model_path}: {e}")
            self.start_epoch = 0
            return False

    def _set_start_epoch(self, checkpoint: dict, model_path: str) -> None:
        """Decide which epoch a resumed run counts from.

        Three cases, and the middle one is a real distinction rather than a special case.
        A checkpoint saved partway through a run is being *resumed*, so the counter picks up
        where it left off. A pretraining or finetuning checkpoint is weights being carried
        into a NEW run, so the counter restarts; `bridge/infra/data/storage_interface.py`
        writes those under `models/pretraining/`, which is where the substring match comes
        from. A checkpoint predating the `epoch` key cannot say, so it starts at 0.

        The guard used to read `"pretraining" not in path or "finetuning" not in path`,
        which is true for every path a checkpoint can realistically have, since one can
        rarely contain both words. It admitted everything, and an unconditional
        `self.start_epoch = 0` underneath undid whatever it decided anyway. This keyed on
        `model_path`, the file actually being loaded, rather than on
        `training_config.checkpoint_path`, so a direct `load_model` call resumes too.

        The substring match is over the whole path, not the filename, because the marker
        sits in a directory component. That is fragile in one direction worth knowing: an
        artifacts directory named `finetuning_runs/` makes every checkpoint under it look
        like a transfer checkpoint and silently restart the counter.
        """
        if "epoch" not in checkpoint:
            self.logger.warning("Checkpoint doesn't contain epoch information, starting from 0")
            self.start_epoch = 0
        elif "pretraining" in model_path or "finetuning" in model_path:
            self.start_epoch = 0
            self.logger.info(
                "%s is a pretraining or finetuning checkpoint, so its weights seed a new "
                "run and the epoch counter starts from 0.",
                model_path,
            )
        else:
            self.start_epoch = checkpoint["epoch"] + 1
            self.logger.info(f"Resuming training from epoch {self.start_epoch}")

    def _warn_on_phoneme_table_drift(self, checkpoint: dict, model_path: str) -> None:
        """Warn if the checkpoint was trained against a different phoneme feature table.

        Phoneme row ids and feature indices are positions in ``phonreps.csv``. Editing that
        file relabels them, and nothing else notices: no parameter shape changes, and the
        derived feature matrix is a non-persistent buffer, so ``load_state_dict`` succeeds
        with ``strict=True``. Warn rather than raise: the weights still run, they just may
        no longer mean what they did, and that is the caller's judgement to make.
        """
        saved = getattr(checkpoint.get("model_config"), "vocab", None)
        # Specs written before the field carry no fingerprint; pickle restores __dict__
        # verbatim, so the attribute can be absent rather than None.
        recorded = getattr(saved, "phon_table_fingerprint", None)
        current = self.model.phon_table_fingerprint
        if recorded is not None and recorded != current:
            self.logger.warning(
                "Phoneme feature table mismatch: %s was trained against table %s but "
                "phonreps.csv now hashes to %s. Phoneme ids may no longer mean what they "
                "did when these weights were trained.",
                model_path,
                recorded,
                current,
            )

    def transfer_partial_model_parameters(
        self, pretrained_model_path: str, module_prefixes: list[str]
    ):
        checkpoint = torch.load(pretrained_model_path, weights_only=False)
        # Transferring a phonological module across a relabelled feature table is the
        # silent-corruption case the fingerprint exists for: shapes still match, so
        # load_state_dict succeeds and nothing else would notice.
        self._warn_on_phoneme_table_drift(checkpoint, pretrained_model_path)
        pretrained_state = checkpoint["model_state_dict"]
        filtered_state = {
            k: v
            for k, v in pretrained_state.items()
            if any(k.startswith(prefix) for prefix in module_prefixes)
        }

        model_dict = self.model.state_dict()
        model_dict.update(filtered_state)
        self.model.load_state_dict(model_dict)

        new_state = self.model.state_dict()
        for key, pretrained_weight in filtered_state.items():
            assert torch.equal(new_state[key], pretrained_weight), (
                f"Weight transfer failed for {key}"
            )
