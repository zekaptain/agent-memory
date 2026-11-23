-- Recall by words reads an index over memories(content, key). Migration 0001 built it with
-- the defaults of the fts extension, whose `ignore` pattern drops every character that is
-- not a letter. A number was therefore no word at all: a query of `0008` or `9494` found
-- nothing, `migration 0009` scored a memory about migration 0008 as it scored one about
-- 0009, and `sha256` found a memory about `sha1`.
--
-- The index is built again with digits kept, over the memories already written. A word with
-- a digit in it is now one word, so `python` no longer finds `python3`. Recall names the
-- index it read: `fts_bm25_v2` for this one, `fts_bm25_v1` for the one before it.
PRAGMA drop_fts_index('memories');
PRAGMA create_fts_index('memories', 'memory_id', 'content', 'key',
    ignore = '(\\.|[^a-z0-9])+', incremental = 1);
