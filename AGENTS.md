# Instructions for oMLX

These rules are for coding agents that work in this repository. The guide for people is [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md). Read it before you open a pull request.

These rules apply to the lines you add or change. Older code in this repository does not always follow them, and it is not a model to copy.

## Commands

- `make dev` installs the server, the web UI and the dev tools, then builds the native custom kernels. On a Mac without full Xcode, use `make dev-no-kernels`.
- `python -m pytest tests/test_<area>.py` runs the tests for the code you changed. Run those first, then `python -m pytest`. The default run skips tests marked `slow` and `integration`.
- `make web` rebuilds the web UI CSS and normalizes the translation files. Run it after you edit a template, JavaScript or `apps/omlx-web/omlx_web/i18n/en.json`.
- `make app` builds the macOS app. Run it when you change `apps/omlx-mac/`.

## Layout

- `omlx/` is the server package. It runs without `apps/` and then serves the API only.
- `apps/omlx-web/` is the web UI. It reaches the server only through the admin API.
- `apps/omlx-mac/` is the SwiftUI app. It starts the server as a subprocess and uses the same admin API.
- `omlx/patches/` holds runtime patches for the pinned mlx-lm and mlx-vlm. A fix for their behavior goes there.

## Before you write code

- Read the callers and the tests of the code you change before you edit it.
- Find the cause first. Fix the place where the behavior first goes wrong, not the place where the symptom shows.
- Do not ship a fix for a cause you have not confirmed. A change that might help does not belong in the diff.
- Make the smallest change that fixes the cause. Reuse the code path that already exists.
- Do not fix a bug by turning a feature off, skipping it for one model, or adding a switch that disables it.
- Do not fix a bug by tuning a constant, such as a chunk size, a retry count, a timeout or a buffer size, unless you have confirmed that the constant is the cause.
- Do not add a setting, a fallback, a retry or a cooldown as extra defense around a fix. Do not add a guard for a state that cannot occur.
- Do not add an `OMLX_*` environment variable that switches behavior. If the user needs a toggle, make it a setting with an entry in the settings UI.
- Do not rewrite a hot path for a small gain.
- Do not remove an existing recovery or fallback branch before you read the commit that added it.
- Do not mix unrelated cleanup into the change.
- CONTRIBUTING.md asks for an issue or a discussion before a change over 1,000 lines or a new subsystem. Tell the user before you start one.

## Comments

- Do not add a comment that restates the code or explains your change. Keep a comment only for behavior that an external dependency forces, or for an invariant the code cannot check. Keep it to one or two lines.
- The comments already in a file are not a model to copy. Match the file in naming and idiom, not in comment density.
- Do not write "do not remove" or "keep in sync" in a comment. Enforce that constraint with a test or an assert.
- Do not mention the conversation, the request or the review in code, names or messages.
- Add a docstring only to a public function or class.
- Use ASCII punctuation in comments and docstrings.

## Code

- Put imports at the top of the file. An optional dependency goes there too, inside `try` and `except ImportError`.
- Validate input where it enters, such as an API request, a config file or the CLI. Inside, do not check again. Write no `None` check, `hasattr` or `getattr` default for a value the caller always sets.
- Do not catch an exception to hide it. Write no `except: pass`, and no `except Exception` around code that cannot raise.
- Do not use `# noqa`, `# type: ignore`, `cast` or `Any` to get a line you wrote past the linter or the type checker. Fix the line.
- Return early instead of nesting.
- Do not add a wrapper with one caller, a helper that only forwards a call, or a second copy of a helper that exists.
- When you replace an internal function, move its callers and delete the old one. Leave no compatibility layer and no parallel old and new path.
- Keep one rule or value in one place.
- Leave no `TODO` or `FIXME` marker, commented-out code, unused import, debug print or scratch file in the diff.
- Do not reformat lines you did not change.
- Do not copy code or text from another project unless its license allows it. Keep the license notice and add the source to an Acknowledgements section in the pull request.

## Tests

- A bug fix gets one regression test where a test is practical. It fails without the fix and passes with it.
- Put the test in the existing test file for that code, and reuse its fixtures and helpers. Do not create a test file when one already covers the module.
- Assert what a caller can observe, with the exact value. Do not assert only that something is set or was called, and do not copy the implementation into the test.
- Do not add tests for trivial behavior, or several cases that check the same thing.
- Do not skip, loosen or delete an existing test to make your change pass. Do not add `sleep` or retries.
- A bug fix that has to rework the test harness or restructure how modules import each other is too big. Look for a smaller one.

## Pull requests

- Search open pull requests for the same fix before you start. If one exists, tell the user and stop.
- Keep one problem in one pull request.
- A performance change needs numbers from `main` and from the change, on the same Mac, model and settings.
- A UI change needs before and after screenshots of the real screen.
- Do not commit work logs, personal test notes or benchmark scripts.

## Pull request text

Start the description from `.github/pull_request_template.md` and keep its headings.

The first section, `TL;DR`, belongs to the user.

- Never write this section.
- Never suggest wording for it, draft it for the user to edit, or rewrite what the user wrote.
- Never fill it from the diff, the commits or the rest of the description.
- You may translate the user's text into English, sentence for sentence. Add nothing and drop nothing. If the user wrote it in English, paste it unchanged.
- If the user gives you no text for this section, do not open the pull request.
- Never tick a box in the checklist. The user ticks them.

The maintainer closes a pull request without review when this section is missing, empty or written by an AI.

You may write the other sections. Under Validation, list what was run, with the Mac and the model, and what remains untested. Leave out test counts, lint results and CI status. Put numbers in a table and add no diagram.

### Examples

User: Open the PR for me.
Agent: I need your TL;DR first. Write one to three sentences on what this change fixes and why. Any language works, and I can translate it. This project does not accept an AI-written TL;DR.

User: Just write the TL;DR, I will check it.
Agent: I cannot write or suggest that section. Tell me in your own words and I will put your text in.

User: 캐시 히트인데 TTFT가 두 배로 늘어나는 문제를 고쳤습니다.
Agent: Here is your text in English, sentence for sentence. "I fixed the problem where TTFT doubles on a cache hit." I added nothing.

User: Make my TL;DR sound better.
Agent: I cannot edit that section. I can translate it as written, or you can change it yourself.
