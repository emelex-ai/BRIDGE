import json
import logging
from collections.abc import Iterator
from pathlib import Path

import torch

from bridge.application.training.ortho_metrics import calculate_orth_metrics
from bridge.application.training.phon_metrics import calculate_phon_metrics
from bridge.core.phonreps import load_phoneme_table
from bridge.domain.data import BridgeDataset
from bridge.domain.datamodels import EncodingComponent, TrainingConfig, TrainingEvent
from bridge.domain.model import Model
from bridge.domain.model.model import PATHWAY_INPUTS, WRITES_ORTH, WRITES_PHON

# Per-step results: loss tensors + scalar metrics + the JSON-encoded `word`.
type MetricsDict = dict[str, torch.Tensor | float | str]


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
        self.model = model
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
        # Nothing here touches the garbage collector. A `gc.collect()` every ten steps cost
        # a measured 17% of every epoch and reclaimed nothing (RSS across an epoch was flat
        # with and without it), because the ~768k objects it rescanned are the
        # process-lifetime pronunciation lexicon cache and can never be freed. Removing it
        # was the whole win. Freezing the heap was tried in its place and is not here
        # either: it bought +0.025 s on a 2.9 s epoch against a 0.149 s run-to-run spread,
        # and in exchange moved every object alive at construction, the caller's included,
        # into a generation the collector never examines. A caller cycle held across
        # `TrainingPipeline(...)` then leaked for the process lifetime, measured at 32 MB,
        # with nothing anywhere to undo it. A library does not get to decide that about
        # someone else's heap.

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

    @property
    def phon_reps(self) -> torch.Tensor:
        """The phonetic-feature block, on whatever device the model is on right now.

        Derived rather than stored, for the same reason `device` is. Snapshotting it in
        `__init__` left a model moved afterwards, which decision 0012 makes legal,
        comparing against a table on the old device, and `cdist` raised.
        `load_phoneme_table` is cached per device, so this costs a dict lookup.
        """
        return load_phoneme_table(device=self.device).phonetic_features

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
            # Placed at the point of use, not at construction. The dataset builds its
            # encodings on the process device and the model goes wherever the caller put
            # it, so something has to reconcile the two; doing it here rather than by
            # moving the model in `__init__` keeps the caller's placement authoritative
            # (decision 0012) and still holds when the model moves mid-run.
            # `BridgeEncoding.to` returns self when the device already matches.
            batch = dataset[sub_slice].to(self.device)
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

    def train_steps(
        self,
        dataset: BridgeDataset,
        batch_slices: list[slice],
        epoch: int = 0,
        calculate_metrics: bool = False,
    ) -> Iterator[TrainingEvent]:
        """Take one optimizer step per slice, yielding a record after each.

        The library owns the step; the caller owns the loop. ``docs/decisions/0006``
        drew that seam and then shipped both halves anyway, so the library still decided
        the train/validation split, the batch size, the per-epoch shuffle, the progress
        bar, the epoch aggregate and the timings. All of that is experiment policy, and
        it is the caller's now. See ``docs/decisions/0013``.

        ``batch_slices`` is whatever partition the caller wants; nothing here assumes the
        slices are contiguous, ordered, disjoint, or drawn from a training split. A caller
        wanting validation runs the same slices under ``torch.no_grad()`` with the model
        in ``eval()``, which is what the deleted ``validate_single_epoch`` did.

            model.eval()
            with torch.no_grad():
                rows = [pipeline.single_step(ds, s, True) for s in val_slices]
        """
        self.model.train()
        for step, batch_slice in enumerate(batch_slices):
            metrics = self.single_step(dataset, batch_slice, calculate_metrics)
            yield TrainingEvent(phase="train", epoch=epoch, step=step, metrics=metrics)

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
