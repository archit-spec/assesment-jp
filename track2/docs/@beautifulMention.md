# @beautifulMention

This run prioritizes Track B (SFT) and Track C (retrieval), with Track A pretraining intentionally skipped.

## Sources Mentioned
- `deepwiki_scripts`: https://github.com/archit-athena/deepwiki-scripts
- `commits_training`: https://github.com/archit-athena/commits-training
- `gist_deepwiki_query`: https://gist.github.com/archit-spec/572bf2ff4ce37cdf4c5606b0d3083ef5
- `gist_coding_agent_minimal`: https://gist.github.com/archit-spec/b5cdcc71d8a2699b74e13481c924b783
- `gist_commit_breakdown`: https://gist.github.com/archit-spec/00053c2a3693500028388ebe3014e68a
- `gist_patch_to_toolcall`: https://gist.github.com/archit-spec/1983d1a2eddc130316d15bdc9d1bb13c
- `gist_toolcall_naturalizer`: https://gist.github.com/archit-spec/8a59e09cfe2d6fff41c041ea7b78f292
- `cpt_hyperswitch_token_aware`: https://huggingface.co/datasets/archit11/hyperswitch-token-aware-cpt-fixed
- `deepwiki_16k`: https://huggingface.co/datasets/archit11/deepwiki-16k
- `issue_pr_new2`: https://huggingface.co/datasets/archit11/new2
- `issue_pr_filenames`: https://huggingface.co/datasets/archit11/hyperswitch-filenames
- `agent_data_collection_code_feedback`: https://huggingface.co/datasets/neulab/agent-data-collection/viewer/code_feedback

## How They Are Used
- Issue/PR/commit style data -> SFT categories aligned to evaluator: `explain`, `complete`.
- Patch/tool-call style data -> converted into `complete` plans and explanation-style supervision.
- Code/doc/query style data -> SFT + retrieval pairs (`docstring`, `improve`, `bugfix`, `unit_test`).

## Output Summary
- SFT pairs generated: `500`
- Retrieval pairs generated: `350`

## Source Load Status
- `deepwiki_scripts` (repo): loaded_rows=0, extracted_records=0
- `commits_training` (repo): loaded_rows=0, extracted_records=0
- `gist_deepwiki_query` (gist): loaded_rows=0, extracted_records=0
- `gist_coding_agent_minimal` (gist): loaded_rows=0, extracted_records=0
- `gist_commit_breakdown` (gist): loaded_rows=1, extracted_records=1
- `gist_patch_to_toolcall` (gist): loaded_rows=0, extracted_records=0
- `gist_toolcall_naturalizer` (gist): loaded_rows=0, extracted_records=0
- `cpt_hyperswitch_token_aware` (hf): loaded_rows=0, extracted_records=0
- `deepwiki_16k` (hf): loaded_rows=0, extracted_records=0
- `issue_pr_new2` (hf): loaded_rows=10, extracted_records=10
- `issue_pr_filenames` (hf): loaded_rows=1, extracted_records=1
- `agent_data_collection_code_feedback` (hf): loaded_rows=1, extracted_records=1
- `local_corpus` (local): loaded_rows=300, extracted_records=300
