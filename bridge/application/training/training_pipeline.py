import gc
import json
import logging
import random
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
from bridge.domain.model.model import PATHWAY_INPUTS, WRITES_ORTH, WRITES_PHON

# Per-step results: loss tensors + scalar metrics + the JSON-encoded `word`.
type MetricsDict = dict[str, torch.Tensor | float | str]
# Per-epoch results: tensors and floats only. `word` is dropped during
# accumulation, and timing values are added as floats.
type NumericMetrics = dict[str, torch.Tensor | float]

min_interval = 1


class TrainingPipeline:
    """Runs optimizer steps over a dataset and reports what happened, one event at a time.

    The library owns the step; the caller owns the loop. Nothing here writes checkpoints,
    logs to a file, or uploads anything: every moment of a run arrives as a
    :class:`~bridge.domain.datamodels.TrainingEvent`, and what to do with one is the
    caller's decision. See ``docs/decisions/0006``.
    """

    def __init__(
        self,
        model: Model,
        training_config: TrainingConfig,
        dataset: BridgeDataset,
    ):
        self.logger = logging.getLogger(__name__)
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
                # the periodic collection this used to run from ~109 ms to ~188 ms.
                tokenizer=self.dataset.tokenizer,
            )
        self.model = model
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
        self.train_slices, self.val_slices = self.create_data_slices()
        self.phon_reps = load_phoneme_table(device=self.device).phonetic_features

        # The pronunciation lexicon is a process-lifetime cache, so the ~768k objects it
        # allocates are immortal and every generational collection rescans them to free
        # nothing. Freezing moves everything allocated so far into a generation the
        # collector skips, which takes `gc.collect()` from a measured 155 ms to 0.0 ms.
        # This replaces a `gc.collect()` every ten steps that cost a measured 17% of every
        # epoch; an RSS control across an epoch showed it reclaiming nothing. Objects
        # allocated after this point are still collected normally.
        gc.freeze()

        self.start_epoch = 0
        if training_config.checkpoint_path:
            self.load_model(training_config.checkpoint_path)

    @property
    def device(self) -> torch.device:
        """Where the model is, asked rather than told.

        A stored snapshot of ``device_manager.device`` plus a ``model.to(...)`` in
        ``__init__`` silently undid the caller's own placement: a model explicitly moved to
        CUDA came back on CPU, all 169 parameters, and the run trained an order of
        magnitude slower with nothing but an INFO line to say so. That is the failure
        ``docs/decisions/0007`` fixed inside :class:`Model`, reintroduced one layer up. The
        pipeline follows the model now. A caller who wants the process device writes
        ``model.to(device_manager.device)``, which is one visible line.
        """
        return self.model.device

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
        """Run the configured pathway over one batch.

        Which tensors a pathway consumes comes from ``PATHWAY_INPUTS`` rather than being
        written out per pathway. Four hand-written branches spelled the same mapping four
        times, and the baseline-capture harness spells it a fifth, so renaming one keyword
        meant finding every copy or the golden master would keep recording the old
        convention.
        """
        pathway = self.training_config.training_pathway
        component = {"orth": orthography, "phon": phonology}
        kwargs: dict[str, torch.Tensor] = {}
        for modality, side in PATHWAY_INPUTS[pathway]:
            part = component[modality]
            kwargs[f"{modality}_{side}_input"] = getattr(part, f"{side}_input_ids")
            kwargs[f"{modality}_{side}_pad_mask"] = getattr(part, f"{side}_pad_mask")
        return self.model(task=pathway, **kwargs)

    def compute_loss(
        self,
        logits: dict[str, torch.Tensor],
        orthography: EncodingComponent,
        phonology: EncodingComponent,
    ) -> dict[str, torch.Tensor]:
        pathway = self.training_config.training_pathway
        vocab = self.model.model_config.vocab
        losses: dict[str, torch.Tensor] = {}

        if pathway in WRITES_PHON:
            losses["phon_loss"] = torch.nn.functional.cross_entropy(
                logits["phon"], phonology.phon_targets, ignore_index=vocab.phon_pad_id
            )

        if pathway in WRITES_ORTH:
            # Teacher forcing: decoder position i is scored against the token at i+1. The
            # character tokenizer lays each sequence out as
            #     enc = [LANG, BOS, ...chars, EOS, PAD...]
            #     dec = [LANG, BOS, ...chars,      PAD...]
            # so the encoder ids shifted left by one give the next token for every decoder
            # position, at the same width. The `[BOS] -> first character` pair this creates
            # is what generation needs: `orthography_decoder_loop` seeds the same prefix and
            # the token it samples next must be the word's first character. See
            # docs/decisions/0004-orthographic-teacher-forcing-alignment.md.
            orth_target = orthography.enc_input_ids[:, 1:]
            self._check_orth_target_width(logits["orth"], orth_target)
            losses["orth_loss"] = torch.nn.functional.cross_entropy(
                logits["orth"], orth_target, ignore_index=vocab.orth_pad_id
            )

        if not losses:
            raise ValueError(f"No loss configured for training_pathway={pathway!r}")

        # The pathway decides how many terms there are; summing the ones present replaces
        # a four-branch cascade over (orth_loss, phon_loss) presence.
        return {"loss": torch.stack(list(losses.values())).sum(), **losses}

    @staticmethod
    def _check_orth_target_width(orth_logits: torch.Tensor, orth_target: torch.Tensor) -> None:
        """Fail with both shapes named when the target and the logits disagree.

        Left to ``cross_entropy`` this reads ``Expected target size [8, 8], got [8, 7]``,
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
        pathway = self.training_config.training_pathway
        metrics: dict[str, float] = {}
        if pathway in WRITES_PHON:
            metrics.update(
                calculate_phon_metrics(
                    logits,
                    phonology,
                    self.phon_reps,
                    phon_pad_id=self.model.model_config.vocab.phon_pad_id,
                )
            )
        if pathway in WRITES_ORTH:
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
        """Run one optimizer step over one slice, in ``num_chunks`` accumulated sub-batches.

        One path, not two. ``num_chunks=1`` is a single sub-slice covering the whole batch,
        which is exactly what the separate fast path did; keeping both meant 62 lines of
        loop-carried bookkeeping shadowing 41 lines that did the same work.

        Reported losses are the sum over sub-batches, unscaled. Only the gradient is divided
        by ``num_chunks``, so accumulating changes what a step costs in memory rather than
        what it optimizes.
        """
        num_chunks = self.training_config.num_chunks or 1
        sub_slices = self._create_sub_slices(batch_slice, num_chunks=num_chunks)
        if not sub_slices:
            raise ValueError(f"batch_slice {batch_slice} is empty; there is nothing to step over.")

        if self.model.training:
            self.optimizer.zero_grad()

        totals: dict[str, torch.Tensor] = {}
        for sub_slice in sub_slices:
            batch = dataset[sub_slice]
            orthography, phonology = batch.orthographic, batch.phonological
            logits = self.forward(orthography, phonology)
            losses = self.compute_loss(logits, orthography, phonology)

            if self.model.training:
                (losses["loss"] / num_chunks).backward()

            for key, value in losses.items():
                totals[key] = totals[key] + value if key in totals else value

        if self.model.training:
            self.optimizer.step()
            self.optimizer.zero_grad()

        # Detached before they leave this method. These are the tensors `backward()` just
        # ran on, so handing them out live means every `TrainingEvent` a caller keeps pins
        # an autograd graph: measured at +581 MB of steady-state RSS inside the loop, and
        # +527 MB per epoch, unbounded, for a caller who keeps the stream to plot a loss
        # curve. Values are identical to 8 decimals either way. `.detach()` rather than
        # `.item()`, which would add a device sync per metric per step on CUDA.
        metrics: MetricsDict = {key: value.detach() for key, value in totals.items()}

        # Metrics come from the last sub-batch. With one chunk that is the whole batch.
        if calculate_metrics:
            metrics.update(self.compute_metrics(logits, orthography, phonology))

        metrics["word"] = json.dumps(dataset.words[batch_slice])
        return metrics

    @staticmethod
    def _create_sub_slices(batch_slice: slice, num_chunks: int) -> list[slice]:
        """Split a slice into smaller slices."""
        start, stop = batch_slice.start, batch_slice.stop
        size = stop - start
        chunk_size = max(1, size // num_chunks)

        return [
            slice(start + i, min(start + i + chunk_size, stop)) for i in range(0, size, chunk_size)
        ]

    @staticmethod
    def _accumulate(total: NumericMetrics, metrics: MetricsDict) -> NumericMetrics:
        """Add one step's numeric metrics into a running total, dropping the string fields."""
        numeric: NumericMetrics = {
            key: value for key, value in metrics.items() if not isinstance(value, str)
        }
        if not total:
            return numeric
        for key, value in numeric.items():
            total[key] = total[key] + value
        return total

    @staticmethod
    def _summarize(
        total: NumericMetrics, steps: int, elapsed: float, prefix: str
    ) -> NumericMetrics:
        """Mean the accumulated metrics, add the two timings, prefix every key.

        One definition. The four hand-written copies this replaces had already drifted:
        three computed ``time_per_epoch`` as elapsed seconds *times* the step count, which
        is not a duration, and the fourth subtracted correctly. A single epoch record
        therefore mixed a correct ``train_time_per_epoch`` with a multiplied ``valid_`` one.
        """
        summary: NumericMetrics = {key: value / steps for key, value in total.items()}
        summary["time_per_step"] = elapsed / steps
        summary["time_per_epoch"] = elapsed
        return {prefix + str(key): value for key, value in summary.items()}

    def _progress(self, slices: list[slice], desc: str) -> Iterator[tuple[int, slice]]:
        """Iterate slices behind a tqdm bar, yielding ``(step, slice)``.

        The postfix is refreshed by :meth:`_show`, which the caller invokes with the metrics
        it just produced. Splitting it this way keeps the throttle in one place instead of
        once per epoch method.
        """
        self._bar = tqdm(slices, desc=desc, mininterval=min_interval)
        self._last_update = time.time()
        yield from enumerate(self._bar)

    def _show(self, metrics: MetricsDict) -> None:
        """Refresh the progress bar postfix, at most once per ``min_interval`` seconds."""
        now = time.time()
        if now - self._last_update <= min_interval:
            return
        self._bar.set_postfix(
            {key: f"{value:.4f}" for key, value in metrics.items() if not isinstance(value, str)}
        )
        self._last_update = now

    def _evaluate(
        self, dataset: BridgeDataset, slices: list[slice], prefix: str, desc: str
    ) -> NumericMetrics:
        """Run one no-grad pass over ``slices``, returning the mean metrics, ``prefix``ed.

        Shared by validation and test, which differed only in the dataset, the slice list
        and the prefix. They were 46 and 42 lines agreeing on 30 of them.
        """
        self.model.eval()
        start = time.time()
        total: NumericMetrics = {}
        with torch.no_grad():
            for _step, batch_slice in self._progress(slices, desc):
                metrics = self.single_step(
                    dataset, batch_slice, self.training_config.compute_metrics
                )
                self._show(metrics)
                total = self._accumulate(total, metrics)
        return self._summarize(total, len(slices), time.time() - start, prefix)

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
        for step, batch_slice in self._progress(self.train_slices, f"Training Epoch {epoch + 1}"):
            metrics = self.single_step(
                self.dataset, batch_slice, self.training_config.compute_metrics
            )
            self._show(metrics)
            yield TrainingEvent(phase="train", epoch=epoch, step=step, metrics=metrics)

    def validate_single_epoch(self, epoch: int) -> NumericMetrics:
        """Score the validation partition, returning ``valid_``-prefixed mean metrics."""
        return self._evaluate(
            self.dataset, self.val_slices, "valid_", f"Validating Epoch {epoch + 1}"
        )

    def test_single_epoch(self, epoch: int) -> NumericMetrics:
        """Score the held-out test set, returning ``test_``-prefixed mean metrics."""
        if self.test_dataset is None:
            raise ValueError("Test dataset not provided in the configuration.")
        test_slices = [
            slice(i, min(i + self.training_config.batch_size_train, len(self.test_dataset)))
            for i in range(0, len(self.test_dataset), self.training_config.batch_size_train)
        ]
        return self._evaluate(self.test_dataset, test_slices, "test_", f"Testing Epoch {epoch + 1}")

    def run_train_val_loop(self, num_epochs: int | None = None) -> Iterator[TrainingEvent]:
        """Train, validating each epoch, yielding a record at every step and boundary.

        The convenience loop, and a thin one: everything it does is available separately as
        :meth:`train_steps`, :meth:`validate_single_epoch`, :meth:`test_single_epoch` and
        :meth:`_shuffle_training_partition`. A caller who wants a different loop writes it
        out of those rather than passing flags into this one.

        It writes no checkpoints and logs nothing. When to save, where, under what name, and
        what to record are all the caller's, from the stream:

            for event in pipeline.run_train_val_loop(num_epochs=3):
                if event.phase == "train" and event.step % 100 == 0:
                    pipeline.save_checkpoint(runs / f"step_{event.step}.pth", event.epoch)
                elif event.phase == "epoch":
                    pipeline.save_checkpoint(runs / f"epoch_{event.epoch}.pth", event.epoch)

        Four phases arrive, and the ``epoch`` one is always last for its epoch. Its metrics
        are the merged aggregate, so a caller who only wants epoch rows filters on that
        phase and is otherwise unchanged.

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
                step_total = self._accumulate(step_total, event.metrics)
                yield event
            if steps:
                epoch_metrics.update(
                    self._summarize(step_total, steps, time.time() - start, "train_")
                )

            if self.val_slices:
                metrics = self.validate_single_epoch(epoch)
                epoch_metrics.update(metrics)
                yield TrainingEvent(phase="validation", epoch=epoch, metrics=dict(metrics))
            if self.test_dataset:
                metrics = self.test_single_epoch(epoch)
                epoch_metrics.update(metrics)
                yield TrainingEvent(phase="test", epoch=epoch, metrics=dict(metrics))

            yield TrainingEvent(phase="epoch", epoch=epoch, metrics=dict(epoch_metrics))

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

        It returns the path for the same reason it uploads nowhere. Copying the file to
        object storage is one more thing the caller decides, and reading a bucket name out
        of the environment made a successful ``torch.save`` raise ``KeyError`` afterwards.

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
        return destination

    def load_model(self, model_path: str) -> None:
        """Restore weights, optimizer state and the epoch counter from a checkpoint.

        Raises rather than reporting failure. A blanket ``except Exception: return False``
        turned a corrupt file, a missing one, a shape mismatch and an unpickling error into
        one return value, and ``__init__`` did not check it, so a failed resume started a
        fresh run from random weights with nothing but a log line to say so. A caller who
        wants best-effort resume writes the ``try`` themselves.
        """
        checkpoint = torch.load(model_path, weights_only=False)
        self._warn_on_phoneme_table_drift(checkpoint, model_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._set_start_epoch(checkpoint, model_path)

    def _set_start_epoch(self, checkpoint: dict, model_path: str) -> None:
        """Decide which epoch a resumed run counts from.

        Three cases, and the middle one is a real distinction rather than a special case.
        A checkpoint saved partway through a run is being *resumed*, so the counter picks up
        where it left off. A pretraining or finetuning checkpoint is weights being carried
        into a NEW run, so the counter restarts. A checkpoint predating the `epoch` key
        cannot say, so it starts at 0.

        The guard used to read `"pretraining" not in path or "finetuning" not in path`,
        which is true for every path a checkpoint can realistically have, since one can
        rarely contain both words. It admitted everything, and an unconditional
        `self.start_epoch = 0` underneath undid whatever it decided anyway. This keys on
        `model_path`, the file actually being loaded, rather than on
        `training_config.checkpoint_path`, so a direct `load_model` call resumes too.

        The substring match is over the whole path, not the filename, because the marker
        usually sits in a directory component. That is fragile in one direction worth
        knowing: an artifacts directory named `finetuning_runs/` makes every checkpoint
        under it look like a transfer checkpoint and silently restart the counter.
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
    ) -> None:
        """Copy the modules named by ``module_prefixes`` out of another checkpoint."""
        checkpoint = torch.load(pretrained_model_path, weights_only=False)
        # Transferring a phonological module across a relabelled feature table is the
        # silent-corruption case the fingerprint exists for: shapes still match, so
        # load_state_dict succeeds and nothing else would notice.
        self._warn_on_phoneme_table_drift(checkpoint, pretrained_model_path)
        pretrained_state = checkpoint["model_state_dict"]
        filtered_state = {
            key: value
            for key, value in pretrained_state.items()
            if any(key.startswith(prefix) for prefix in module_prefixes)
        }

        model_dict = self.model.state_dict()
        model_dict.update(filtered_state)
        self.model.load_state_dict(model_dict)
