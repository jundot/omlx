{
  description = "oMLX - LLM inference server optimized for Apple Silicon";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "aarch64-darwin";
      pkgs = import nixpkgs { inherit system; };
      python = pkgs.python312;
      version = pkgs.lib.removeSuffix "\""
        (pkgs.lib.removePrefix "__version__ = \""
          (pkgs.lib.trim (builtins.readFile ./omlx/_version.py)));
    in
    {
      packages.${system}.default = python.pkgs.buildPythonApplication {
        pname = "omlx";
        inherit version;
        pyproject = true;
        src = self;
        build-system = with python.pkgs; [
          setuptools
          wheel
        ];

        dependencies = with python.pkgs; [
          fastapi
          itsdangerous
          jsonschema
          mlx
          mlx-lm
          pillow
          requests
          uvicorn
        ];
        dontCheckRuntimeDeps = true;

        meta = {
          description = "LLM inference server optimized for Apple Silicon";
          homepage = "https://github.com/jundot/omlx";
          license = pkgs.lib.licenses.asl20;
          maintainers = with pkgs.lib.maintainers; [ dzmitry-lahoda ];
          platforms = [ system ];
          mainProgram = "omlx";
        };
      };

      devShells.${system}.default = pkgs.mkShell {
        packages = [
          python
          pkgs.uv
        ];
      };
    };
}
