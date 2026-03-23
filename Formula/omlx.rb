class Omlx < Formula
  desc "LLM inference server optimized for Apple Silicon"
  homepage "https://github.com/jundot/omlx"
  url "https://github.com/jundot/omlx/archive/refs/tags/v0.2.20.tar.gz"
  sha256 "c9a557568ad4a62bbb9a9a2324d84e0687259b77ab84eb2f85e92b273e7a9456"
  license "Apache-2.0"

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
