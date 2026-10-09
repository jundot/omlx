# Developer shortcuts. The server itself builds with plain pip and needs none of these.
PYTHON ?= python3
# setuptools, cmake, and nanobind pins from [build-system]; the kernels must use MLX's nanobind.
KERNEL_BUILD_DEPS = $(shell $(PYTHON) -c "import tomllib; \
	reqs = tomllib.load(open('pyproject.toml', 'rb'))['build-system']['requires']; \
	print(' '.join(repr(r) for r in reqs if r.startswith(('setuptools', 'cmake', 'nanobind'))))")

.PHONY: install install-no-kernels mcp dev dev-no-kernels kernels clean-kernels web app

# Editable install of the server and the web UI, then the native custom kernels.
install: install-no-kernels
	$(MAKE) kernels

# Editable install without the kernels, for machines without full Xcode. The affected
# model families then fall back to much slower generic paths.
install-no-kernels:
	$(PYTHON) -m pip install -e .

# Add MCP (Model Context Protocol) support to the editable install.
mcp:
	$(PYTHON) -m pip install -e ".[mcp]"

# Same as install, with dev tools.
dev: dev-no-kernels
	$(MAKE) kernels

dev-no-kernels:
	$(PYTHON) -m pip install -e ".[dev]"

# Rebuild every native custom kernel in place from scratch, then check that each one loads.
# Needs full Xcode for the Metal compiler.
kernels: clean-kernels
	$(PYTHON) -m pip install --quiet $(KERNEL_BUILD_DEPS)
	OMLX_WITH_CUSTOM_KERNEL=1 $(PYTHON) setup.py build_ext --inplace --force
	$(PYTHON) -c "from omlx.custom_kernels import native_kernel_status; \
	failed = {k: v['import_error'] for k, v in native_kernel_status().items() if not v['available']}; \
	print(failed or 'All native custom kernels are available.'); raise SystemExit(1 if failed else 0)"

# Delete compiled custom kernels and their CMake build trees.
clean-kernels:
	rm -f omlx/custom_kernels/*/*.so omlx/custom_kernels/*/*.dylib omlx/custom_kernels/*/*.metallib
	rm -rf build/temp.*/omlx.custom_kernels.* build/lib*/omlx/custom_kernels

# Rebuild the web UI CSS and normalize its translation files.
web:
	cd apps/omlx-web && $(PYTHON) build_css.py && $(PYTHON) normalize_i18n.py

# Build a runnable macOS app bundle with freshly compiled native custom kernels.
app:
	apps/omlx-mac/Scripts/build.sh release --with-custom-kernel
