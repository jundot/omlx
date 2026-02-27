class Omlx < Formula
  desc "LLM inference server optimized for Apple Silicon"
  homepage "https://github.com/jundot/omlx"
  url "https://github.com/jundot/omlx/archive/refs/tags/v0.1.9.tar.gz"
  sha256 "c09902e397da4566e2c2c455fa586034accad97b99923ec384bfa903268529c8"
  license "Apache-2.0"

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

    # Install package - pip resolves ALL deps from pyproject.toml
    system libexec/"bin/pip", "install", buildpath

    bin.install_symlink Dir[libexec/"bin/omlx"]
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/omlx --version")
  end
end
