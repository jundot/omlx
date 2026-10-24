# Experimental JANGH2 contribution

The optional [JANGH2 review package](../contrib/jangh2/README.md) contains packed
Metal expert execution, strict checkpoint loading, SSD-backed PLE, opt-in grouped
prefill/device caching and qualification/benchmark tools.

This is an **RFC**, not a current-main runtime feature. Applying the patch does
not modify ordinary `omlx serve`. Its serving adapter was exercised in a legacy
PR #364 environment. See the [maintainer notes](../contrib/jangh2/UPSTREAM.md)
before attempting to integrate those private hooks with newer schedulers.

CPU tests:

```sh
cd contrib/jangh2
python -m unittest discover -s tests -v
```

Apple Metal execution and full-model tests require the supported Mac environment.
The optional default-command installer can reuse an already working environment
without changing Homebrew, model files, dependencies or the macOS application.
