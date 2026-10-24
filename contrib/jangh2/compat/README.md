# Legacy reproduction details (not current-main changes)

Reference checkout: PR #364 head
`a77d410f3f5948854bb42e5363239883282ac1d4` (`0.3.0rc1`).

The user had a modified local checkout, not a pristine reference installation.
Known local adaptations were:

- In `EnginePool._load_engine`, initialise `effective_type = entry.engine_type`
  before dispatch. The inspected reference still uses the name without assigning.
- The user's local `get_engine()` lacked its post-load return. It was repaired.
  The inspected reference head already contains
  `return self._entries[model_id].engine`; do not insert a second return there.
- Replace the old Transformers range with `>=5.14.0,<5.18.0` and replace the old
  MLX-VLM git dependency with `mlx-vlm==0.7.8` in that local pyproject.
- Actual reported runtime versions included MLX 0.32.4 and Transformers 5.17.0.
  The entire current environment has NOT been captured from the user's machine.
  These facts are not a complete lockfile. Use `capture_environment.py` on the
  working Mac rather than reconstructing all transitive dependencies by guesswork.
- Earlier troubleshooting patched a Homebrew PLE alias and linked a Qwen4 model
  module, then upgraded MLX-VLM. Those historical changes are not needed as new
  upstream core edits and are not re-applied by this package.

`pr364-dependencies-and-dispatch.patch` records the minimal remaining reference
source changes for review. It is not applied automatically. The dependency diff
is experimental legacy compatibility, NOT a recommendation to downgrade or
replace current-main dependencies. Prefer the already-working virtualenv.

The scope of this contribution excludes unrelated PR #364 commits, downloaded
model weights and any private local edits not present in the supplied artifacts.
