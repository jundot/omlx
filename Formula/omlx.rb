class Omlx < Formula
  include Language::Python::Virtualenv

  desc "LLM inference server optimized for Apple Silicon"
  homepage "https://github.com/jundot/omlx"
  url "https://github.com/jundot/omlx/archive/refs/tags/v0.1.7.tar.gz"
  sha256 "e69e3df7539fa59ec3b5cd18374247daae00ad768f4c61d6b2b1de8c5f4a1555"
  license "Apache-2.0"

  depends_on "python@3.11"
  depends_on :macos
  depends_on arch: :arm64

  def install
    venv = virtualenv_create(libexec, "python3.11")
    venv.pip_install buildpath
    bin.install_symlink Dir[libexec/"bin/omlx"]
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/omlx --version")
  end
end
