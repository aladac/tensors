# Refactoring Plan

Generated: 2026-02-12
Scope: `tensors/` (all modules) + `tests/`

## Summary

11 issues found: 3 high priority, 5 medium priority, 3 low priority.

## High Priority

### [ ] Extract common HTTP request pattern in api.py
- **Location**: `api.py:38-108`
- **Problem**: `fetch_civitai_model_version`, `fetch_civitai_model`, and `fetch_civitai_by_hash` share nearly identical try/except/error-handling blocks (3x duplication).
- **Impact**: Changing error handling or adding retry logic requires editing 3+ places. Easy to introduce inconsistencies (e.g., `fetch_civitai_model_version` lacks the Progress spinner that the other two have).
- **Action**: Extract a `_api_get(url, api_key, console, spinner_msg=None) -> dict | None` helper that handles GET, 404 check, error printing, and optional Progress spinner. Each fetch function becomes a one-liner calling this helper.

### [ ] Eliminate module-level console singleton in cli.py
- **Location**: `cli.py:67`
- **Problem**: `console = Console()` at module level is used by 6 helper functions (`_output_info_json`, `_save_metadata`, `_resolve_by_hash`, `_resolve_by_model_id`, `_prepare_download_dir`, `_display_download_info`) via closure rather than parameter passing.
- **Impact**: Impossible to inject a test console into helpers, prevents output capture in tests, creates hidden coupling. Commands pass `console` to API/display functions but helpers silently use the global.
- **Action**: Add `console: Console` parameter to all helper functions. Pass the module-level console from command functions. This makes the dependency explicit and testable.

### [ ] Remove or use unused `safetensors` dependency
- **Location**: `pyproject.toml:8`
- **Problem**: `safetensors>=0.4.0` is listed as a runtime dependency but never imported anywhere. The code does manual binary parsing in `safetensor.py`.
- **Impact**: Unnecessary install bloat. Users install a native extension library they don't need. Confusing for contributors.
- **Action**: Remove `safetensors` from `[project.dependencies]` since manual parsing is the intended approach.

## Medium Priority

### [ ] Deduplicate output dict construction in cli.py
- **Location**: `cli.py:126-133` and `cli.py:154-161`
- **Problem**: `_output_info_json` and `_save_metadata` construct the same dict structure independently.
- **Impact**: Adding a new field to info output requires editing two places.
- **Action**: Extract `_build_info_dict(file_path, sha256_hash, local_metadata, civitai_data) -> dict` and call it from both functions.

### [ ] Extract API timeout and chunk size constants
- **Location**: `api.py:43,70,97,175` and `safetensor.py:67`
- **Problem**: `timeout=30.0` appears 4 times in api.py. Chunk sizes (`1024 * 1024` in api.py, `1024 * 1024 * 8` in safetensor.py) are inline magic numbers.
- **Impact**: Changing timeout or chunk size requires hunting through code.
- **Action**: Add `API_TIMEOUT = 30.0`, `DOWNLOAD_CHUNK_SIZE = 1024 * 1024` to config.py or at module top. Add `HASH_CHUNK_SIZE = 1024 * 1024 * 8` to safetensor.py constants section.

### [ ] Rename `_format_size` and `_format_count` to public API
- **Location**: `display.py:23,32`
- **Problem**: `_format_size` is imported and used by `cli.py:36` as a cross-module API but has a private underscore prefix. Same for `_format_count` which could be useful externally.
- **Impact**: Violates Python naming convention. Underscore-prefixed names signal "don't import this" but the codebase does.
- **Action**: Rename to `format_size` and `format_count`. Update all imports.

### [ ] Deduplicate Progress spinner setup in api.py
- **Location**: `api.py:61-66,88-93,166-171`
- **Problem**: The `Progress(SpinnerColumn(), TextColumn(...), console=..., transient=True)` pattern is repeated 3 times with identical configuration.
- **Impact**: Changing spinner appearance requires 3 edits.
- **Action**: Extract `_spinner(console, description)` context manager helper or a factory function. This pairs well with the HTTP helper extraction in High Priority #1.

### [ ] Add shared console fixture to tests
- **Location**: `tests/test_tensors.py:284,291,298,...`
- **Problem**: `Console(force_terminal=True, width=80)` is created ~15 times across test methods.
- **Impact**: Changing test console config requires editing every test.
- **Action**: Add a `console` fixture to `conftest.py` and use it across all test classes.

## Low Priority

### [ ] Replace hardcoded command list in main()
- **Location**: `cli.py:403`
- **Problem**: `arg not in ("info", "search", "get", "dl", "download", "config")` is a manually maintained list of known commands for legacy invocation detection.
- **Impact**: Adding a new command requires remembering to update this list, or legacy mode will swallow it.
- **Action**: Derive the command list from `app.registered_commands` or Typer's command registry dynamically.

### [ ] Strengthen display function tests
- **Location**: `tests/test_tensors.py:279-385`
- **Problem**: Most display tests only assert "should not raise" with no output verification.
- **Impact**: Tests won't catch regressions in output format or content.
- **Action**: Use `Console(record=True)` to capture output, then assert key strings appear in `console.export_text()`.

### [ ] Add tests for untested pure functions
- **Location**: `api.py:111-142,145-150,187-199,202-209`
- **Problem**: `_build_search_params`, `_filter_results`, `_setup_resume`, `_get_dest_from_response` have no test coverage. These are pure/near-pure functions that are easy to test.
- **Impact**: Logic errors in search parameter building or resume setup won't be caught.
- **Action**: Add direct unit tests for each function. These don't need HTTP mocking since they're pure logic.

## Notes

- High Priority #1 (HTTP helper) and Medium #6 (spinner dedup) should be done together since the extracted helper will naturally absorb the spinner logic.
- High Priority #2 (console singleton) should be done before Medium #7 (test fixture) since making console a parameter enables proper test injection.
- Medium #4 (output dict) is a quick win that can be done independently.
- Start with High Priority #3 (remove unused dep) as it's the simplest and lowest risk change.
