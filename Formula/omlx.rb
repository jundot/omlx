class Omlx < Formula
  desc "LLM inference server optimized for Apple Silicon"
  homepage "https://github.com/jundot/omlx"
  url "https://github.com/jundot/omlx/archive/refs/tags/v0.5.3.tar.gz"
  sha256 "11d91196f376231d89da4500e0ee54b97c4de4601df546d6aacd03df6d40ea0a"
  license "Apache-2.0"

  head "https://github.com/jundot/omlx.git", branch: "main"

  option "with-custom-kernel",
         "Build native custom kernels for GLM-5.2, MiniMax M3 and Qwen3.5/3.6 acceleration"
  option "with-grammar", "Install xgrammar for structured output (requires torch, ~2GB)"

  depends_on "rust" => :build
  depends_on arch: :arm64
  depends_on :macos
  depends_on "python@3.11"

  # macOS 27 beta's `strip` corrupts dynamic offsets in Mach-O libraries
  # (llvm/llvm-project#203678). Skip Homebrew's post-install clean pass over
  # the venv so it never runs `strip` on the compiled dylibs.
  on_macos do
    skip_clean "libexec" if MacOS.version >= "27"
  end

  # mlx-audio pins mlx-lm==0.31.1 which conflicts with omlx's git-pinned
  # mlx-lm. Fetch source separately so we can patch the pin before install.
  resource "mlx-audio" do
    url "https://github.com/Blaizzy/mlx-audio.git",
      revision: "51753266e0a4f766fd5e6fbc46652224efc23981"
  end

  # Kokoro's English G2P path uses misaki + spaCy. Bundle the spaCy
  # language model so the first TTS request does not download into the
  # Homebrew venv at runtime.
  resource "en-core-web-sm" do
    url "https://github.com/explosion/spacy-models/releases/download/" \
        "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
    sha256 "1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85"
  end

  service do
    run [opt_bin/"omlx", "serve"]
    keep_alive true
    working_dir var
    log_path var/"log/omlx.log"
    error_log_path var/"log/omlx.log"
    environment_variables PATH: std_service_path_env
  end

  def install
    # Create venv with pip so dependency resolution works properly
    system "python3.11", "-m", "venv", libexec

    # Build native extensions from source with headerpad so Homebrew can
    # rewrite Mach-O install names to absolute Cellar/opt paths. Rust/maturin
    # extension builds (cohere_melody) need the linker flag via RUSTFLAGS;
    # C/C++ extension builds use LDFLAGS.
    ENV.append "LDFLAGS", "-Wl,-headerpad_max_install_names"
    ENV.append "RUSTFLAGS", "-C link-arg=-Wl,-headerpad_max_install_names"

    no_binary = "cohere_melody,pydantic-core,rpds-py,tiktoken"
    pip_flags = []
    if MacOS.version >= "27"
      # macOS 27's dyld requires the LC_SYMTAB string pool to start on an
      # 8-byte boundary; prebuilt Rust wheels aligned to 4 bytes fail dlopen
      # with "mis-aligned LINKEDIT string pool". Build them from source, and
      # keep Cargo/maturin's release stripping off so the beta's broken
      # `strip` (llvm/llvm-project#203678) never touches the fresh dylibs.
      no_binary += ",tokenizers"
      ENV["CARGO_PROFILE_RELEASE_STRIP"] = "false"
      ENV["MATURIN_STRIP"] = "false"
      # Pip reuses locally built wheels even under --no-binary, so a wheel
      # cached before the strip guards existed stays corrupted. Bypass the
      # cache entirely.
      pip_flags << "--no-cache-dir"
    end

    # Every pip step must share these flags; a later step without --no-binary
    # (e.g. mlx-audio) can clobber a source-built package with a prebuilt
    # wheel that fails dlopen on macOS 27 (#2110).
    pip_install = [libexec/"bin/pip", "install", *pip_flags, "--no-binary", no_binary]

    if build.with?("custom-kernel")
      kernel_sources = [
        buildpath/"omlx/custom_kernels/glm_moe_dsa/csrc",
        buildpath/"omlx/custom_kernels/minimax_m3/csrc",
        buildpath/"omlx/custom_kernels/qwen35_prefill/csrc",
      ]
      unless kernel_sources.all?(&:directory?)
        odie "--with-custom-kernel requires oMLX custom kernel sources; use --HEAD or a release that includes them"
      end

      ENV["OMLX_WITH_CUSTOM_KERNEL"] = "1"
      # Pin CMake to the venv's Python; its default discovery can pick a
      # newer unlinked system Python instead. setup.py forwards CMAKE_ARGS
      # to the kernel builds.
      ENV.append "CMAKE_ARGS", "-DPython_EXECUTABLE=#{libexec}/bin/python"
    end

    # Install omlx (with optional grammar extra for structured output)
    install_spec = build.with?("grammar") ? "#{buildpath}[grammar]" : buildpath.to_s
    system(*pip_install, install_spec)

    if build.with?("custom-kernel")
      # Run from libexec so buildpath's raw omlx/ source tree doesn't shadow
      # the compiled package in the venv's site-packages.
      Dir.chdir(libexec) do
        system libexec/"bin/python", "-c", <<~PYTHON
          from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast
          from omlx.custom_kernels.minimax_m3 import fast as minimax_fast
          from omlx.custom_kernels.qwen35_prefill import fast as qwen35_fast
          assert glm_fast.is_native_available(), glm_fast.import_error()
          assert minimax_fast.is_native_available(), minimax_fast.import_error()
          assert qwen35_fast.is_native_available(), qwen35_fast.import_error()
        PYTHON
      end
    end

    # Install mlx-audio with patched mlx-lm pin to avoid version conflict
    resource("mlx-audio").stage do
      inreplace "pyproject.toml", '"mlx-lm==0.31.1"', '"mlx-lm>=0.31.1"'
      system(*pip_install, ".[all]")
    end

    # Install the spaCy English model required by misaki for Kokoro TTS.
    # Homebrew's cached resource path is hash-prefixed, which pip rejects
    # as an invalid wheel filename. Copy it back to the canonical basename.
    spacy_model_wheel = buildpath/"en_core_web_sm-3.8.0-py3-none-any.whl"
    cp resource("en-core-web-sm").cached_download, spacy_model_wheel
    system libexec/"bin/pip", "install", "--no-deps",
           spacy_model_wheel
    system libexec/"bin/python", "-c",
           "import spacy; spacy.load('en_core_web_sm')"

    # python-multipart is declared in omlx's [audio] extra, not in mlx-audio
    system(*pip_install, "python-multipart>=0.0.5")

    bin.install_symlink Dir[libexec/"bin/omlx"]
  end

  # Both fixups below must run in post_install rather than install because
  # Homebrew's post-install "Cleaning" step rewrites Mach-O install names
  # and deletes every dist-info/RECORD file in the keg as part of its
  # relocation pass. Anything patched inside `def install` is either wiped
  # or invalidated before the user sees it.
  def post_install
    return if build.without?("grammar") && build.without?("custom-kernel")

    python = libexec/"bin/python"
    site = Utils.safe_popen_read(python, "-c",
                                 "import site; print(site.getsitepackages()[0])").chomp
    patch_xgrammar(python, site) if build.with?("grammar")
    fix_custom_kernel_rpaths(python, site) if build.with?("custom-kernel")
  end

  # Patch the macOS arm64 xgrammar wheel so its native binding loads.
  # The 0.1.32+ wheel ships libxgrammar_bindings.dylib with
  # @rpath/libtvm_ffi.dylib but no LC_RPATH pointing at where tvm_ffi
  # installs its native lib, and the dist-info is missing a RECORD
  # entry for the dylib so tvm_ffi's manifest-based lookup fails.
  # Both manifest as RuntimeError("Cannot find library: ...") at
  # `import xgrammar`, which crashes /admin/api/grammar/parsers and
  # hides the Reasoning Parser dropdown. Tracking upstream:
  # jundot/omlx#1005.
  def patch_xgrammar(python, site)
    ohai "Patching xgrammar macOS arm64 wheel"
    tvmlib = Utils.safe_popen_read(python, "-c",
      "import os, tvm_ffi; print(os.path.join(os.path.dirname(tvm_ffi.__file__), 'lib'))").chomp
    dylib = "#{site}/xgrammar/libxgrammar_bindings.dylib"
    dist_dirs = Dir["#{site}/xgrammar-*.dist-info"]

    ohai "  site=#{site}"
    ohai "  tvmlib=#{tvmlib}"
    ohai "  dylib=#{dylib} (exists? #{File.exist?(dylib)})"
    ohai "  dist-info=#{dist_dirs.inspect}"

    odie "xgrammar dylib not found at #{dylib}" unless File.exist?(dylib)
    odie "xgrammar dist-info not found under #{site}" if dist_dirs.empty?

    # Patch 1: add tvm_ffi/lib to the dylib's rpath, then re-codesign so
    # macOS will load the modified dylib.
    rpaths = Utils.safe_popen_read("/usr/bin/otool", "-l", dylib)
    if rpaths.include?(tvmlib)
      ohai "  rpath already points at tvm_ffi/lib"
    else
      ohai "  adding rpath -> #{tvmlib}"
      system "/usr/bin/install_name_tool", "-add_rpath", tvmlib, dylib
      system "/usr/bin/codesign", "--force", "--sign", "-", dylib
    end

    # Patch 2: ensure RECORD lists the dylib so tvm_ffi's manifest-based
    # lookup finds it. Brew's clean pass already deleted every RECORD by
    # the time post_install runs, so we always (re)create one.
    record = "#{dist_dirs.first}/RECORD"
    if File.exist?(record) && File.read(record).include?("libxgrammar_bindings.dylib")
      ohai "  RECORD already lists the dylib"
    else
      ohai "  writing dylib entry to #{record}"
      File.open(record, "a") { |f| f.puts "xgrammar/libxgrammar_bindings.dylib,," }
    end

    # Verify the patch took. Failing here is much less confusing than
    # the user discovering it later via a 500 from the admin route.
    ohai "  verifying import xgrammar..."
    system python, "-c", "import xgrammar; print('xgrammar import OK')"
  end

  # The custom kernel extensions reference @rpath/libmlx.dylib but their
  # only link-time libmlx rpath points into pip's isolated build env, which
  # is dead after install. The import check in `def install` still passes
  # because dyld resolves the dependency against the already-loaded libmlx
  # by install name; the post-install "Cleaning" pass then rewrites
  # libmlx's LC_ID_DYLIB to an absolute Cellar path, which breaks that
  # match, so the kernels silently fail to dlopen at runtime and prefill
  # falls back to the slow path (issue #2233). Stamp the real mlx lib dir
  # as an rpath after the clean pass and re-verify from the final state.
  def fix_custom_kernel_rpaths(python, site)
    ohai "Adding mlx rpath to custom kernel binaries"
    mlx_lib = Utils.safe_popen_read(python, "-c",
      "import os, mlx.core; print(os.path.join(os.path.dirname(mlx.core.__file__), 'lib'))").chomp
    odie "mlx lib dir not found at #{mlx_lib}" unless File.directory?(mlx_lib)
    binaries = Dir["#{site}/omlx/custom_kernels/*/{_ext*.so,lib*_kernel_ops.dylib}"]
    odie "no custom kernel binaries under #{site}/omlx/custom_kernels" if binaries.empty?

    binaries.each do |lib|
      if Utils.safe_popen_read("/usr/bin/otool", "-l", lib).include?(mlx_lib)
        ohai "  #{File.basename(lib)}: mlx rpath already present"
        next
      end
      ohai "  adding rpath to #{File.basename(lib)}"
      system "/usr/bin/install_name_tool", "-add_rpath", mlx_lib, lib
      system "/usr/bin/codesign", "--force", "--sign", "-", lib
    end

    ohai "  verifying custom kernel imports..."
    system python, "-c", <<~PYTHON
      from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast
      from omlx.custom_kernels.minimax_m3 import fast as minimax_fast
      from omlx.custom_kernels.qwen35_prefill import fast as qwen35_fast
      assert glm_fast.is_native_available(), glm_fast.import_error()
      assert minimax_fast.is_native_available(), minimax_fast.import_error()
      assert qwen35_fast.is_native_available(), qwen35_fast.import_error()
    PYTHON
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/omlx --version")
    system libexec/"bin/python", "-c",
           "import spacy; spacy.load('en_core_web_sm')"
    if build.with?("custom-kernel")
      system libexec/"bin/python", "-c", <<~PYTHON
        from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast
        from omlx.custom_kernels.minimax_m3 import fast as minimax_fast
        from omlx.custom_kernels.qwen35_prefill import fast as qwen35_fast
        assert glm_fast.is_native_available(), glm_fast.import_error()
        assert minimax_fast.is_native_available(), minimax_fast.import_error()
        assert qwen35_fast.is_native_available(), qwen35_fast.import_error()
      PYTHON
    end
  end
end
