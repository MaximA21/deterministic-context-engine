---
title: "Boilerplate: structure-aware pinning for JSON/SQL/Config"
labels: scorer, good-first-issue
---

## Summary

Add structure-aware detection that pins boilerplate content (JSON schemas, SQL CREATE TABLE, config blocks) at a minimum priority floor so it survives compaction when referenced by the conversation.

## Motivation

Boilerplate like database schemas, API response formats, and config files is often critical context that gets evicted because it scores low on keyword/goal alignment. These chunks should be pinned when the conversation references their domain.

## Proposed Approach

1. Detect structured content types using regex patterns:
   - JSON: opening `{` with quoted keys
   - SQL DDL: `CREATE TABLE`, `ALTER TABLE`, etc.
   - Config: `[section]` headers, `key = value` patterns (TOML/INI)
   - YAML: indented `key:` patterns
2. When a structured chunk is detected and the conversation references related entities (table names, config keys, JSON fields), pin it with a minimum priority floor
3. Integrate with the existing `EntityExtractor` to cross-reference entity mentions

## Acceptance Criteria

- [ ] Detects JSON, SQL DDL, TOML/INI, and YAML boilerplate
- [ ] Pinned boilerplate survives compaction when referenced
- [ ] Unreferenced boilerplate is still eligible for eviction
- [ ] `niah_boilerplate.py` benchmark shows improvement
- [ ] Unit tests for each content type detection
