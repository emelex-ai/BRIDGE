# 0013. The caller owns the loop, in fact and not only in principle

Status: Accepted
Date: 2026-09-19
Supersedes part of [0006](0006-the-caller-owns-the-training-loop.md)

## Context

`docs/decisions/0006` drew the seam between the step and the loop, and then shipped both
halves in the same package. The library still decided:

- the train/validation split, as a contiguous cut at `train_test_split` of the dataset,
  which for an alphabetically sorted lexicon puts every late-alphabet word in validation;
- the batch size, twice, through `batch_size_train` and `batch_size_val`;
- whether and how to shuffle between epochs, by calling `random.seed` on the **global**
  `random` module and reshuffling the training partition in place;
- how many epochs to run;
- that there is a progress bar, on stdout, in a library;
- what an epoch aggregate is, including `time_per_step` and `time_per_epoch`;
- which file is the test set, via `test_data_path`, resolved against BRIDGE's own install
  root rather than the caller's working directory.

None of that requires access to the model. All of it is experiment policy, and a repo whose
role is to be the importable computational core of somebody else's experiment should not be
making those choices for them.

The scaffolding also carried a disproportionate share of the defects. Four hand-written
copies of the epoch aggregate had drifted, three computing `time_per_epoch` as elapsed
seconds *times* the step count. The shared progress bar, held on the instance, crashed tqdm
at interpreter teardown when two public iterations interleaved. `_shuffle_training_partition`
reseeded process-global RNG from inside a library call.

Measured before deciding: a workflow that tokenizes, builds a model, runs all five pathways
forward, generates on all five and decodes executes **zero** lines of `training_pipeline.py`.
Nothing outside the training half imports into it; the dependency runs one way.

## Decision

**`TrainingPipeline` keeps the step and loses the loop.** What remains:

| kept | why |
|---|---|
| `forward` | maps `PATHWAY_INPUTS` to the model's kwargs, which is the model's contract |
| `compute_loss`, `_check_orth_target_width` | the teacher-forcing shift and the two `ignore_index` choices are model contract, not scoring preference (decisions 0004, 0005) |
| `compute_metrics` | the alignment conventions behind it fail silently when re-derived; see Consequences |
| `single_step`, `_create_sub_slices` | the seam itself, including gradient accumulation and the detach that keeps autograd graphs out of the caller's hands |
| `train_steps(dataset, batch_slices, epoch, calculate_metrics)` | one step per slice, one `TrainingEvent` each. Takes the partition rather than owning it |
| `save_checkpoint`, `load_model`, `_warn_on_phoneme_table_drift` | what belongs in a bundle, and the fingerprint check, are privileged knowledge |

`transfer_partial_model_parameters` went too, in the same pass. It had no caller of any
kind: the one test that named it read the method's source with `inspect.getsource` and
asserted a substring appeared in it. Filtering a `state_dict` by module prefix and calling
`load_state_dict` is six lines of public torch API, and the only privileged part, the
phoneme-table drift check, is reachable through `load_model`. That check is now pinned by
driving a real drifted checkpoint through `load_model` and asserting the warning, which
was verified to fail when the call is removed.

Removed: `run_train_val_loop`, `_evaluate`, `validate_single_epoch`, `test_single_epoch`,
`_accumulate`, `_summarize`, `_postfix` and every `tqdm` bar, `create_data_slices`,
`_shuffle_training_partition`, and the test-dataset construction in `__init__`.

`TrainingConfig` drops `num_epochs`, `batch_size_train`, `batch_size_val`,
`train_test_split`, `test_data_path`, `seed`, `shuffle_each_epoch` and `compute_metrics`,
keeping the six fields that describe the step: `learning_rate`, `weight_decay`,
`num_chunks`, `training_pathway`, `checkpoint_path`, `model_artifacts_dir`.

**`TrainingEvent` and `TrainingPhase` stay exported.** They are the shared vocabulary for a
run rather than a thing only this library produces. `train_steps` emits `train` events; a
caller's own loop emits `validation`, `test` and `epoch` ones. Keeping all four phases is
what lets a downstream logger consume any BRIDGE run without inventing its own record type.

## Consequences

`tqdm` leaves the dependency list. Install is torch, pydantic, pandas and numpy.

`TrainingPipeline` goes from 606 lines to 416, and the four-copy aggregate, the shared bar
and the global reseed go with the loop rather than needing individual fixes.

A caller now writes the loop, which is the point but is also more code than before:

```python
slices = [slice(i, i + 32) for i in range(0, cutpoint, 32)]
for epoch in range(pipeline.start_epoch, num_epochs):
    dataset.shuffle(cutpoint)
    for event in pipeline.train_steps(dataset, slices, epoch, calculate_metrics=True):
        ...
```

`start_epoch` is therefore a value the caller reads rather than one the library consumes,
and `BridgeDataset.shuffle` is called by the caller rather than from inside an epoch.

**The metrics stay, and the reason is worth recording** because it is the exception to the
rule this record applies everywhere else. `phon_metrics` and `ortho_metrics` need no
privileged access and could be written downstream, which by this record's own test would
put them out. They stay because three of their conventions fail *silently* when re-derived,
measured on a real batch:

| convention | the obvious guess | correct | what the guess does |
|---|---|---|---|
| phonological pad id | `orth_pad_id`, 2 | `phon_pad_id`, 35 | keeps 100% of positions, scoring 44% padding as content |
| orthographic target | `dec_input_ids` | `enc_input_ids[:, 1:]` | same width, different content: scores positions the loss never trained |
| special-token columns | drop `len(SPECIAL_TOKENS)`, 5 | `[:, :base_dim]`, 31 | drops one real phonetic feature |

Each produces a plausible number rather than an error, and two of the three have already
been got wrong in this repo (issue #225, decision 0005). Absorbing a trap that fails loudly
is not worth a library's weight; absorbing one that fails quietly is.
