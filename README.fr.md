<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/icon-rounded-dark.svg" width="140">
    <source media="(prefers-color-scheme: light)" srcset="docs/images/icon-rounded-light.svg" width="140">
    <img alt="oMLX" src="docs/images/icon-rounded-light.svg" width="140">
  </picture>
</p>

<h1 align="center">oMLX</h1>
<p align="center"><b>Inférence LLM, optimisée pour votre Mac</b><br>Batching continu et cache KV à plusieurs niveaux, géré directement depuis votre barre de menus.</p>

<p align="center">
<a href="https://www.buymeacoffee.com/jundot"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="40"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License">
  <img src="https://img.shields.io/badge/python-3.10+-green" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/platform-Apple%20Silicon-black?logo=apple" alt="Apple Silicon">
</p>

<p align="center">
  <a href="mailto:junkim.dot@gmail.com">junkim.dot@gmail.com</a> · <a href="https://omlx.ai/me">https://omlx.ai/me</a>
</p>

<p align="center">
  <a href="#installation">Installation</a> ·
  <a href="#démarrage-rapide">Démarrage rapide</a> ·
  <a href="#fonctionnalités">Fonctionnalités</a> ·
  <a href="#modèles">Modèles</a> ·
  <a href="#configuration-cli">Configuration CLI</a> ·
  <a href="https://omlx.ai/benchmarks">Benchmarks</a> ·
  <a href="https://omlx.ai">oMLX.ai</a>
</p>

<p align="center">
  <a href="README.md">English</a> ·
  <a href="README.zh.md">中文</a> ·
  <a href="README.ko.md">한국어</a> ·
  <a href="README.ja.md">日本語</a> ·
  <b>Français</b>
</p>

---

<p align="center">
  <img src="docs/images/omlx_dashboard.png" alt="oMLX Admin Dashboard" width="800">
</p>

> *Chaque serveur LLM que j'ai essayé m'obligeait à choisir entre commodité et contrôle. Je voulais garder des modèles du quotidien en mémoire, permuter automatiquement les plus lourds à la demande, définir des limites de contexte — et tout gérer depuis la barre de menus.*
>
> *oMLX persiste le cache KV sur un niveau chaud en RAM et un niveau froid sur SSD — même quand le contexte change en cours de conversation, tout le contexte passé reste en cache et réutilisable entre les requêtes, rendant les LLM locaux vraiment pratiques pour du vrai travail de code avec des outils comme Claude Code. C'est pour ça que je l'ai construit.*

## Installation

### Application macOS

Téléchargez le `.dmg` depuis les [Releases](https://github.com/jundot/omlx/releases), glissez-le dans Applications, c'est tout. L'application inclut une mise à jour automatique intégrée, les futures mises à jour se font en un clic. À noter que l'application macOS n'installe pas la commande CLI `omlx`. Pour une utilisation en terminal, installez via Homebrew ou depuis les sources.

### Homebrew

```bash
brew tap jundot/omlx https://github.com/jundot/omlx
brew install omlx

# Mettre à jour vers la dernière version
brew update && brew upgrade omlx

# Lancer en service en arrière-plan (redémarre automatiquement en cas de crash)
brew services start omlx

# Optionnel : support MCP (Model Context Protocol)
/opt/homebrew/opt/omlx/libexec/bin/pip install mcp
```

### Depuis les sources

```bash
git clone https://github.com/jundot/omlx.git
cd omlx
pip install -e .          # Core uniquement
pip install -e ".[mcp]"   # Avec support MCP (Model Context Protocol)
```

Nécessite macOS 15.0+ (Sequoia), Python 3.10+, et Apple Silicon (M1/M2/M3/M4).

## Démarrage rapide

### Application macOS

Lancez oMLX depuis votre dossier Applications. L'écran de bienvenue vous guide en trois étapes — répertoire des modèles, démarrage du serveur, et premier téléchargement de modèle. C'est tout. Pour connecter OpenClaw, OpenCode, Codex ou Hermes Agent, voir [Intégrations](#intégrations).

<p align="center">
  <img src="docs/images/Screenshot 2026-02-10 at 00.36.32.png" alt="oMLX Welcome Screen" width="360">
  <img src="docs/images/Screenshot 2026-02-10 at 00.34.30.png" alt="oMLX Menubar" width="240">
</p>

### CLI

```bash
omlx serve --model-dir ~/models
```

Le serveur détecte automatiquement les LLM, VLM, modèles d'embedding et rerankers depuis les sous-répertoires. N'importe quel client compatible OpenAI peut se connecter à `http://localhost:8000/v1`. Une interface de chat intégrée est aussi disponible à `http://localhost:8000/admin/chat`.

### Service Homebrew

Si vous avez installé via Homebrew, vous pouvez lancer oMLX en service géré en arrière-plan :

```bash
brew services start omlx    # Démarrer (redémarre automatiquement en cas de crash)
brew services stop omlx     # Arrêter
brew services restart omlx  # Redémarrer
brew services info omlx     # Vérifier le statut
```

Le service lance `omlx serve` avec les paramètres par défaut (`~/.omlx/models`, port 8000). Pour personnaliser, définissez des variables d'environnement (`OMLX_MODEL_DIR`, `OMLX_PORT`, etc.) ou lancez `omlx serve --model-dir /votre/chemin` une fois pour sauvegarder les paramètres dans `~/.omlx/settings.json`.

Les logs sont écrits à deux endroits :
- **Log du service** : `$(brew --prefix)/var/log/omlx.log` (stdout/stderr)
- **Log du serveur** : `~/.omlx/logs/server.log` (log applicatif structuré)

## Fonctionnalités

Prend en charge les LLM texte, les modèles vision-langage (VLM), les modèles OCR, les embeddings et les rerankers sur Apple Silicon.

### Tableau de bord Admin

Interface web sur `/admin` pour le monitoring en temps réel, la gestion des modèles, le chat, les benchmarks et les réglages par modèle. Disponible en anglais, coréen, japonais, chinois, français et russe. Toutes les dépendances CDN sont incluses pour un fonctionnement entièrement hors-ligne.

<p align="center">
  <img src="docs/images/Screenshot 2026-02-10 at 00.45.34.png" alt="oMLX Admin Dashboard" width="720">
</p>

### Modèles vision-langage (VLM)

Exécutez des VLM avec le même stack de batching continu et cache KV à plusieurs niveaux que les LLM texte. Supporte le chat multi-images, les entrées images en base64/URL/fichier, et l'appel d'outils avec contexte visuel. Les modèles OCR (DeepSeek-OCR, DOTS-OCR, GLM-OCR) sont auto-détectés avec des prompts optimisés.

### Cache KV à deux niveaux (Chaud + Froid)

Gestion de cache KV par blocs inspirée de vLLM, avec partage de préfixe et Copy-on-Write. Le cache opère sur deux niveaux :

- **Niveau chaud (RAM)** : Les blocs fréquemment accédés restent en mémoire pour un accès rapide.
- **Niveau froid (SSD)** : Quand le cache chaud se remplit, les blocs sont déversés sur SSD au format safetensors. À la prochaine requête avec un préfixe correspondant, ils sont restaurés depuis le disque plutôt que recalculés — même après un redémarrage du serveur.

<p align="center">
  <img src="docs/images/omlx_hot_cold_cache.png" alt="oMLX Hot & Cold Cache" width="720">
</p>

### Batching continu

Gère les requêtes concurrentes via le BatchGenerator de mlx-lm. Le nombre maximum de requêtes simultanées est configurable via CLI ou le panneau d'administration.

### Optimisation Claude Code

Support du context scaling pour faire tourner des modèles avec un contexte réduit avec Claude Code. Ajuste les compteurs de tokens reportés pour que l'auto-compactage se déclenche au bon moment, et un keep-alive SSE évite les timeouts de lecture pendant les longs prefills.

### Service multi-modèles

Chargez des LLM, VLM, modèles d'embedding et rerankers sur le même serveur. Les modèles sont gérés via une combinaison de contrôles automatiques et manuels :

- **Éviction LRU** : Les modèles les moins récemment utilisés sont évincés automatiquement quand la mémoire est faible.
- **Chargement/déchargement manuel** : Les badges de statut interactifs dans le panneau d'admin permettent de charger ou décharger des modèles à la demande.
- **Épinglage de modèles** : Épinglez les modèles fréquemment utilisés pour les garder toujours chargés.
- **TTL par modèle** : Définissez un délai d'inactivité par modèle pour le décharger automatiquement après une période sans activité.
- **Contrôle mémoire processus** : La limite mémoire totale (par défaut : RAM système - 8 Go) évite les OOM système.

### Paramètres par modèle

Configurez les paramètres d'échantillonnage, les kwargs du template de chat, le TTL, l'alias du modèle, le type de modèle, et plus encore — directement depuis le panneau d'admin. Les modifications s'appliquent immédiatement sans redémarrage du serveur.

- **Alias de modèle** : définissez un nom personnalisé visible par l'API. `/v1/models` retourne l'alias, et les requêtes acceptent l'alias comme le nom du répertoire.
- **Type de modèle** : forcez manuellement un modèle en LLM ou VLM indépendamment de l'auto-détection.

<p align="center">
  <img src="docs/images/omlx_ChatTemplateKwargs.png" alt="oMLX Chat Template Kwargs" width="480">
</p>

### Chat intégré

Chattez directement avec n'importe quel modèle chargé depuis le panneau d'admin. Supporte l'historique de conversation, le changement de modèle, le mode sombre, l'affichage des raisonnements, et l'upload d'images pour les modèles VLM/OCR.

<p align="center">
  <img src="docs/images/ScreenShot_2026-03-14_104350_610.png" alt="oMLX Chat" width="720">
</p>

### Téléchargement de modèles

Recherchez et téléchargez des modèles MLX depuis HuggingFace directement dans le tableau de bord. Parcourez les fiches modèles, vérifiez les tailles de fichiers, et téléchargez en un clic.

<p align="center">
  <img src="docs/images/downloader_omlx.png" alt="oMLX Model Downloader" width="720">
</p>

### Intégrations

Configurez OpenClaw, OpenCode, Codex, Hermes Agent et Pi directement depuis le tableau de bord en un clic. Aucune édition manuelle de config requise.

<p align="center">
  <img src="docs/images/omlx_integrations.png" alt="oMLX Integrations" width="720">
</p>

### Benchmark de performance

Benchmarking en un clic depuis le panneau d'admin. Mesure le prefill (PP) et la génération de tokens (TG) en tokens par seconde, avec des tests de hit partiel sur le cache de préfixe pour des chiffres réalistes.

<p align="center">
  <img src="docs/images/benchmark_omlx.png" alt="oMLX Benchmark Tool" width="720">
</p>

### Application barre de menus macOS

Application native PyObjC dans la barre de menus (pas Electron). Démarrez, arrêtez et surveillez le serveur sans ouvrir un terminal. Inclut des statistiques de service persistantes (survivent aux redémarrages), un redémarrage automatique en cas de crash, et une mise à jour automatique intégrée.

<p align="center">
  <img src="docs/images/Screenshot 2026-02-10 at 00.51.54.png" alt="oMLX Menubar Stats" width="400">
</p>

### Compatibilité API

Remplacement direct des APIs OpenAI et Anthropic. Supporte les statistiques d'usage en streaming (`stream_options.include_usage`), le thinking adaptatif Anthropic, et les entrées visuelles (base64, URL).

| Endpoint | Description |
|----------|-------------|
| `POST /v1/chat/completions` | Complétion de chat (streaming) |
| `POST /v1/completions` | Complétion de texte (streaming) |
| `POST /v1/messages` | API Messages Anthropic |
| `POST /v1/embeddings` | Embeddings texte |
| `POST /v1/rerank` | Reranking de documents |
| `GET /v1/models` | Lister les modèles disponibles |

### Appel d'outils et sorties structurées

Supporte tous les formats d'appel de fonctions disponibles dans mlx-lm, la validation de schéma JSON, et l'intégration d'outils MCP. L'appel d'outils nécessite que le template de chat du modèle supporte le paramètre `tools`. Les familles de modèles suivantes sont auto-détectées via les parseurs intégrés de mlx-lm :

| Famille de modèles | Format |
|---|---|
| Llama, Qwen, DeepSeek, etc. | JSON `<tool_call>` |
| Série Qwen3.5 | XML `<function=...>` |
| Gemma | `<start_function_call>` |
| GLM (4.7, 5) | XML `<arg_key>/<arg_value>` |
| MiniMax | `<minimax:tool_call>` |
| Mistral | `[TOOL_CALLS]` |
| Kimi K2 | `<\|tool_calls_section_begin\|>` |
| Longcat | `<longcat_tool_call>` |

Les modèles non listés ci-dessus peuvent quand même fonctionner si leur template de chat accepte `tools` et que leur sortie utilise un format XML `<tool_call>` reconnu. Pour le streaming avec outils, le texte de l'assistant est émis de façon incrémentale tandis que les balises de contrôle des appels d'outils sont supprimées du contenu visible ; les appels d'outils structurés sont émis après parsing du tour complet.

## Modèles

Pointez `--model-dir` vers un répertoire contenant des sous-répertoires de modèles au format MLX. L'organisation en deux niveaux (ex. `mlx-community/nom-du-modèle/`) est aussi supportée.

```
~/models/
├── Step-3.5-Flash-8bit/
├── Qwen3-Coder-Next-8bit/
├── gpt-oss-120b-MXFP4-Q8/
├── Qwen3.5-122B-A10B-4bit/
└── bge-m3/
```

Les modèles sont auto-détectés par type. Vous pouvez aussi télécharger des modèles directement depuis le tableau de bord.

| Type | Modèles |
|------|--------|
| LLM | Tout modèle supporté par [mlx-lm](https://github.com/ml-explore/mlx-lm) |
| VLM | Série Qwen3.5, GLM-4V, Pixtral, et autres modèles [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) |
| OCR | DeepSeek-OCR, DOTS-OCR, GLM-OCR |
| Embedding | BERT, BGE-M3, ModernBERT |
| Reranker | ModernBERT, XLM-RoBERTa |

## Configuration CLI

```bash
# Limite mémoire pour les modèles chargés
omlx serve --model-dir ~/models --max-model-memory 32GB

# Limite mémoire au niveau du processus (par défaut : auto = RAM - 8 Go)
omlx serve --model-dir ~/models --max-process-memory 80%

# Activer le cache SSD pour les blocs KV
omlx serve --model-dir ~/models --paged-ssd-cache-dir ~/.omlx/cache

# Définir la taille du cache chaud en mémoire
omlx serve --model-dir ~/models --hot-cache-max-size 20%

# Ajuster le nombre max de requêtes simultanées (par défaut : 8)
omlx serve --model-dir ~/models --max-concurrent-requests 16

# Avec les outils MCP
omlx serve --model-dir ~/models --mcp-config mcp.json

# Endpoint miroir HuggingFace (pour les régions restreintes)
omlx serve --model-dir ~/models --hf-endpoint https://hf-mirror.com

# Authentification par clé API
omlx serve --model-dir ~/models --api-key votre-clé-secrète
# Localhost uniquement : désactivez la vérification via les paramètres globaux du panneau d'admin
```

Tous les paramètres peuvent aussi être configurés depuis le panneau d'admin web sur `/admin`. Les paramètres sont sauvegardés dans `~/.omlx/settings.json`, et les flags CLI ont la priorité.

<details>
<summary>Architecture</summary>

```
Serveur FastAPI (API OpenAI / Anthropic)
    │
    ├── EnginePool (multi-modèles, éviction LRU, TTL, chargement/déchargement manuel)
    │   ├── BatchedEngine (LLMs, batching continu)
    │   ├── VLMEngine (modèles vision-langage)
    │   ├── EmbeddingEngine
    │   └── RerankerEngine
    │
    ├── ProcessMemoryEnforcer (limite mémoire totale, vérifications TTL)
    │
    ├── Scheduler (FCFS, concurrence configurable)
    │   └── mlx-lm BatchGenerator
    │
    └── Stack de cache
        ├── PagedCacheManager (GPU, par blocs, CoW, partage de préfixe)
        ├── Hot Cache (niveau RAM, write-back)
        └── PagedSSDCacheManager (niveau froid SSD, format safetensors)
```

</details>

## Développement

### Serveur CLI

```bash
git clone https://github.com/jundot/omlx.git
cd omlx
pip install -e ".[dev]"
pytest -m "not slow"
```

### Application macOS

Nécessite Python 3.11+ et [venvstacks](https://venvstacks.lmstudio.ai) (`pip install venvstacks`).

```bash
cd packaging

# Build complet (venvstacks + bundle app + DMG)
python build.py

# Ignorer venvstacks (modifications de code uniquement)
python build.py --skip-venv

# DMG uniquement
python build.py --dmg-only
```

Voir [packaging/README.md](packaging/README.md) pour les détails sur la structure du bundle app et la configuration des couches.

## Contribuer

Les contributions sont les bienvenues ! Voir le [Guide de contribution](docs/CONTRIBUTING.md) pour les détails.

- Corrections de bugs et améliorations
- Optimisations de performance
- Améliorations de la documentation

## Licence

[Apache 2.0](LICENSE)

## Remerciements

- [MLX](https://github.com/ml-explore/mlx) et [mlx-lm](https://github.com/ml-explore/mlx-lm) par Apple
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) — inférence de modèles vision-langage sur Apple Silicon
- [vllm-mlx](https://github.com/waybarrios/vllm-mlx) — oMLX a démarré à partir de vllm-mlx v0.1.0 et a considérablement évolué avec le service multi-modèles, le cache KV à plusieurs niveaux, le support VLM avec cache paginé complet, un panneau d'admin, et une app dans la barre de menus macOS
- [venvstacks](https://venvstacks.lmstudio.ai) — couches d'environnements Python portables pour le bundle de l'app macOS
- [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings) — support des modèles d'embedding sur Apple Silicon
- [dflash-mlx](https://github.com/bstnxbt/dflash-mlx) — décodage spéculatif par diffusion par blocs sur Apple Silicon
