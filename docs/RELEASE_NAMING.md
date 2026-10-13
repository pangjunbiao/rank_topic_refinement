# Public release names and saved-run compatibility

The primary report is exposed through `--primary-affinity`, implemented by `primary_affinity_analysis`, and exported under `step08_primary_affinity_analysis`. Its bundle and table names describe the analysis rather than a recipient. The source, CLI help and newly produced reports use the same terminology.

Scientific arrays, masks, fitting vocabulary, model parameters and archived predictions are retained unchanged. Archived input names and metadata remain private evidence excluded from Git. They are not rewritten to manufacture a new provenance record.

The public source edits the teaching analogy in one vendored GSDMM docstring while retaining its license. Its executable AST is unchanged. The saved-run source check recognizes exactly the archived and release hashes for that docstring edit and continues to reject other implementation mismatches. `docs/source_lineage.json` records the archived-to-public source correspondence by hashes and professional public paths.

Renamed reporting modules have new output identities. Their tables and divergence statistics are recomputed from the same frozen inputs, with no retraining or parameter selection. Software validation and saved-array replay do not certify a fresh end-to-end model-training run on arbitrary hardware.
