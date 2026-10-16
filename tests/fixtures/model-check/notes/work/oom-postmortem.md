# Postmortem: ingest pods

## Timeline

09:40 ingest pods entered CrashLoopBackOff, every one OOMKilled.

## Cause

A batch size change made each worker hold 2.3 GiB; the limit was 2 GiB.

## Fix

Raised limits to 3 GiB and capped batch size at 512.
