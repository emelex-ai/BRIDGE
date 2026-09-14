# 0001. Record architecture decisions

Status: Accepted
Date: 2026-08-30

## Context

The repo carries decisions whose reasoning is not recoverable from the code. Why phonology is
shaped the way it is, why a fixture must never be regenerated, why a buffer is non-persistent.
Reading the code tells you what is true, never why, and a future reader cannot tell a
deliberate constraint from an accident.

`docs/architecture.md` describes the current state and is rewritten freely as the system
changes. That makes it the wrong place for history: keeping the reasoning there either freezes
the map or loses the reasoning.

## Decision

Record each significant architecture decision as a numbered file in `docs/decisions/`,
following Michael Nygard's Architecture Decision Record format: Context, Decision,
Consequences.

Records are **immutable and append-only**. A decision that no longer holds is not edited. A new
record supersedes it, and the old one is marked `Superseded by NNNN` so the reasoning that
applied at the time survives.

Where evidence drove the decision, the record carries the actual numbers and names the script
that produced them, per `~/.claude/rules/measurement.md`.

`docs/architecture.md` carries an index of records so their existence and subject are visible
without loading them all.

## Consequences

The map stays current and short. The reasoning stays available and does not rot. A reader
revisiting a decision reads the record first and learns what was known at the time, which is
the information needed to judge whether it still applies.

The cost is discipline: a decision made without a record is a decision whose reasoning is lost,
and nothing enforces writing one.
