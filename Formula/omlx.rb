class Omlx < Formula
  desc "LLM inference server optimized for Apple Silicon"
  homepage "https://github.com/jundot/omlx"
  url "https://github.com/jundot/omlx/archive/refs/tags/v0.1.8.tar.gz"
  sha256 "5f87d17134e22dadf10b59ca0e0b2fa6074db6226898b49eee06da91b3f1a70b"
  license "Apache-2.0"

  depends_on "python@3.11"
  depends_on :macos
  depends_on arch: :arm64

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
