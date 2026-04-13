class Omlx < Formula
  desc "LLM inference server optimized for Apple Silicon"
  homepage "https://github.com/jundot/omlx"
  url "https://github.com/jundot/omlx/archive/refs/tags/v0.3.5.dev1.tar.gz"
  sha256 "c9877d8070fcbe4a98ca687815be79cf9dea5392e74e4e88a4554e563fe7f4d3"
  license "Apache-2.0"

  head "https://github.com/jundot/omlx.git", branch: "main"

  option "with-grammar", "Install xgrammar for structured output (requires torch, ~2GB)"

  depends_on "rust" => :build
  depends_on "python@3.11"
  depends_on :macos
  depends_on arch: :arm64

  # mlx-audio pins mlx-lm==0.31.1 which conflicts with omlx's git-pinned
  # mlx-lm. Fetch source separately so we can patch the pin before install.
  resource "mlx-audio" do
    url "https://github.com/Blaizzy/mlx-audio.git",
      revision: "51753266e0a4f766fd5e6fbc46652224efc23981"
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

    # Build Rust-based packages from source with headerpad to prevent
    # Homebrew dylib ID fixup failure (Mach-O header too small for absolute paths).
    # tokenizers is excluded: its wheel ships a stable-ABI .abi3.so that does
    # not need Homebrew's dylib ID rewrite, and building from source fails on
    # macOS 15+ due to PyO3 linker errors (missing Python symbols at link time).
    ENV.append "LDFLAGS", "-Wl,-headerpad_max_install_names"

    # Install omlx (with optional grammar extra for structured output)
    install_spec = build.with?("grammar") ? "#{buildpath}[grammar]" : buildpath.to_s
    system libexec/"bin/pip", "install", "--no-binary", "pydantic-core,rpds-py,tiktoken", install_spec

    # Install mlx-audio with patched mlx-lm pin to avoid version conflict
    resource("mlx-audio").stage do
      inreplace "pyproject.toml", '"mlx-lm==0.31.1"', '"mlx-lm>=0.31.1"'
      system libexec/"bin/pip", "install", ".[all]"
    end

    # python-multipart is declared in omlx's [audio] extra, not in mlx-audio
    system libexec/"bin/pip", "install", "python-multipart>=0.0.5"

    bin.install_symlink Dir[libexec/"bin/omlx"]
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/omlx --version")
  end
end
