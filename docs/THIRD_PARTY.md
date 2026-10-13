# Third-party implementation provenance

The project uses released neural model modules and installed scientific packages. Attribution and licensing are separate from a manuscript citation file.

| Method | Upstream repository |
|---|---|
| NeuroMax | https://github.com/Fsoft-AIC/NeuroMax |
| EnCOT | https://github.com/manhdo249/EnCOT |
| FASTopic | https://github.com/BobXWu/FASTopic |
| GloCOM | https://github.com/qducnguyen/GloCOM |
| CombinedTM | https://github.com/MilaNLProc/contextualized-topic-models |

Pinned revisions, module paths, byte counts and SHA-256 checksums are recorded in `configs/upstream_neural_sources.json` and `configs/upstream_competitor_sources.json`. The loaders fetch only those files and validate them before use. Do not present their source as original project code.

The vendored GSDMM module preserves the MIT license and Ryan Walker attribution in `src/rank_refinement/vendor/GSDMM_LICENSE.txt`. Keep this notice with redistributed copies. Installed packages and downloaded encoder models retain their own terms. Before a public release or bundling third-party source/weights, review the applicable upstream licenses and required notices.

No new license is assigned to the original project by this preparation. No `CITATION.cff` or fabricated publication reference is included.
