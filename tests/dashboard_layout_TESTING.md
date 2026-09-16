# Dashboard layout interaction checks

`dashboard_layout_release.test.cjs` runs the actual layout JavaScript in jsdom.
It uses fixed, mocked card rectangles. It checks event/state behavior, not browser rendering or visual smoothness.

Install jsdom in a disposable directory, not in the server environment:

```sh
TEST_DEPS=$(mktemp -d)
npm install --prefix "$TEST_DEPS" --no-save --package-lock=false --ignore-scripts jsdom@30.0.1
NODE_PATH="$TEST_DEPS/node_modules" node tests/dashboard_layout_release.test.cjs
```

The test checks no DOM reordering or placement during a drag, the insertion marker,
release persistence, cancellation, restored natural height, and slider preview/commit separation.
It does not load a model or connect to a server. The `--baseline` option reads committed HEAD instead of working-tree files for local RED/GREEN checks.

For manual checks, drag slowly across tile boundaries, scroll while dragging, cancel with Escape, and move the width slider in both directions. Tiles should remain stationary during the gesture and reflow on release. Check Safari and Firefox separately; jsdom does not establish browser compatibility.
