class Omlx < Formula
  desc "LLM inference server optimized for Apple Silicon"
  homepage "https://github.com/jundot/omlx"
  url "https://github.com/jundot/omlx/archive/refs/tags/v0.2.22.tar.gz"
  sha256 "d276415648ff3206ce1e5f2e266fbdbe4497e8ccd312b2b3a5062502be1b577c"
  license "Apache-2.0"

  head "https://github.com/jundot/omlx.git", branch: "main"

  depends_on "rust" => :build
  depends_on "python@3.11"
  depends_on :macos
  depends_on arch: :arm64

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

    # Upgrade pip to ensure modern resolver (handles git deps, etc.)
    system libexec/"bin/pip", "install", "--upgrade", "pip"

    # Build Rust-based packages from source with headerpad to prevent
    # Homebrew dylib ID fixup failure (Mach-O header too small for absolute paths)
    ENV.append "LDFLAGS", "-Wl,-headerpad_max_install_names"
    system libexec/"bin/pip", "install", "--no-binary", "pydantic-core,rpds-py,tiktoken,tokenizers", buildpath

    bin.install_symlink Dir[libexec/"bin/omlx"]
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/omlx --version")
  end
end
