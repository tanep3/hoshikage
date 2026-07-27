# Hoshikage User Manual

English | [日本語](user-manual.md)

Hoshikage manages GGUF models and provides Chat Completions and Responses APIs. An upper agent such as Codex CLI owns tool execution and iteration; Hoshikage owns model execution and protocol translation.

## 1. Installation

### 1.1 Requirements

- Stable Rust toolchain
- llama.cpp runtime bundle for Hoshikage
- GGUF models to serve
- A compatible CUDA environment when using a GPU

```bash
cargo install --path .
hoshikage --version
```

Install the runtime bundle in `llama.cpp` under the operating-system-specific Hoshikage configuration directory by default. Set `HOSHIKAGE_LLAMA_CPP_RUNTIME_DIR` to use another location.

### 1.2 Configuration file

The standard Hoshikage server configuration directory depends on the operating system.

| OS | Hoshikage configuration directory |
|---|---|
| Linux | `~/.config/hoshikage` |
| macOS | `~/Library/Application Support/hoshikage` |
| Windows | `%APPDATA%\hoshikage` |

Hoshikage reads `.env` from this directory. Set `HOSHIKAGE_CONFIG_PATH` to use another file. This is storage for the Hoshikage server itself and its Token administration; it is not the Codex or upper-application configuration location.

```dotenv
HOST=127.0.0.1
PORT=3030
N_CTX=65536
HOSHIKAGE_LLAMA_SERVER_CACHE_TYPE_K=q8_0
HOSHIKAGE_LLAMA_SERVER_CACHE_TYPE_V=q8_0
HOSHIKAGE_LANG=ja
```

Binding to `HOST=127.0.0.1` or `localhost` allows unauthenticated loopback access. Binding to a LAN address or `0.0.0.0` requires Bearer Token authentication.

## 2. Model Management

### 2.1 Registering a model

```bash
hoshikage add /models/gemma4/model.gguf unsloth-gemma4-12b-qat-thinking-off --n-ctx 65536 --thinking-off
hoshikage list --details
```

The Codex compatibility floor is 16K, but 64K is recommended for practical agent workloads that use tools and skills. When a model has no explicit `n_ctx`, Hoshikage uses `N_CTX` from `.env`. Because KV cache usage grows with context, start with `HOSHIKAGE_LLAMA_SERVER_CACHE_TYPE_K/V=q8_0` on 12 GB-class GPUs and verify the actual headroom with the platform GPU monitor.

The Bundle `tool_calling` setting is authoritative. Bundles without this setting default safely to `disabled`. `doctor` reports candidates and inconsistencies but never rewrites a Bundle.

### 2.2 Bundle diagnostics

```bash
hoshikage doctor --model unsloth-gemma4-12b-qat-thinking-off
hoshikage doctor --model unsloth-gemma4-12b-qat-thinking-off --json
```

With `--json`, field names, statuses, IDs, and `message_key` values never change with the display language. Automation should use these values instead of human-readable text.

## 3. Loopback Quick Start

### 3.1 Starting the server

Set `HOST=127.0.0.1` in `.env`, then start Hoshikage.

```bash
hoshikage
curl http://127.0.0.1:3030/health
curl http://127.0.0.1:3030/ready
```

### 3.2 Testing the Responses API

```bash
curl http://127.0.0.1:3030/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model":"unsloth-gemma4-12b-qat-thinking-off","input":"Return exactly OK."}'
```

Loopback access does not require a Token. Use `/health` for process liveness, `/ready` for configuration and runtime readiness, and `/v1/status` for model loading state.

## 4. LAN and Tokens

### 4.1 What a Token is

A Token is a secret passphrase proving that a device is allowed to use Hoshikage over the LAN. It is not an OpenAI API key, ChatGPT login, or model license key. It is unnecessary for a same-machine `127.0.0.1` connection, but required when another computer connects over the LAN.

Any device that knows the Token can request inference from Hoshikage. Treat it like a password and never put it in Git, Model Bundles, issues, logs, or chat. Do not write the Token itself into Codex TOML. The upper application that starts Codex passes it through the process environment variable `HOSHIKAGE_API_KEY`.

### 4.2 Creating a Token on Hoshikage

Log in as an administrator on the Hoshikage server machine and create a named Token for each client. `hoshikage token` is not a remote API; it directly administers the Token store on that machine.

```bash
hoshikage token create codex-desktop
hoshikage token list
```

`token list` prints the name, Token plaintext, public ID, creation time, and update time. It shows all information because it is an administrator tool on the server machine. Do not run it while sharing the screen, collecting terminal output, or where another person can see it. `codex-desktop` is an administrative name, not the Token itself.

```text
codex-desktop	hsk_xxx_xxx	public_id=xxx	created=1780000000	updated=1780000000
```

The Token store is `auth_tokens.json` in the standard Hoshikage server configuration directory. Hoshikage sets and validates owner-only `0600` access on Linux and macOS, and a protected ACL granting full control only to the owner and SYSTEM on Windows. Upper applications and Codex never read this file directly.

### 4.3 Listening on the LAN

Set `.env` in the Hoshikage server location documented in section 1.2, then restart Hoshikage.

```dotenv
HOST=0.0.0.0
PORT=3030
```

This guide uses `192.168.1.50` as the Hoshikage server LAN address. On a Linux or macOS client, run:

```bash
curl http://192.168.1.50:3030/health
```

On Windows, run the following in PowerShell and confirm that `TcpTestSucceeded : True` and health information are returned.

```powershell
Test-NetConnection 192.168.1.50 -Port 3030
Invoke-RestMethod http://192.168.1.50:3030/health
```

If either check fails, inspect the IP address, Hoshikage process, and operating-system firewall. Expose Hoshikage only on a trusted home or organizational LAN. Do not configure WAN port forwarding on the router.

### 4.4 Passing the Token from an upper application

The standard owner of Token delivery is the upper application, such as Yatagarasu, that starts Codex. It puts the selected Token in the child process `HOSHIKAGE_API_KEY` and does not need to register it permanently for the entire operating system.

For a manual check on Linux or macOS, read the Token without leaving it in shell history.

```bash
printf "HOSHIKAGE_API_KEY: "
IFS= read -rs HOSHIKAGE_API_KEY
printf "\n"
export HOSHIKAGE_API_KEY
codex exec --profile hoshikage "Return exactly the word OK."
unset HOSHIKAGE_API_KEY
```

The following example sets it only in the current Windows PowerShell process. The input is visible on screen, so take care around other people and screen sharing.

```powershell
$env:HOSHIKAGE_API_KEY = Read-Host "HOSHIKAGE_API_KEY"
codex exec --profile hoshikage "Return exactly the word OK."
Remove-Item Env:HOSHIKAGE_API_KEY
```

### 4.5 Passing the Token to the Windows Codex app

When launching the Windows Codex app directly without an upper application to inject the Token, register a Windows user environment variable.

1. Search for "environment variables" in the Windows Start menu.
2. Open "Edit environment variables for your account."
3. Under "User variables," select "New."
4. Enter `HOSHIKAGE_API_KEY` as the variable name and the Token plaintext as its value.
5. Close every dialog with OK.
6. Fully quit the running Codex app and start it again.

The following PowerShell command performs the same persistent registration. The Token may remain in PowerShell history, so the graphical procedure above is normally safer.

```powershell
[Environment]::SetEnvironmentVariable("HOSHIKAGE_API_KEY", "<token>", "User")
```

Check whether the variable exists without displaying its value.

```powershell
if ([Environment]::GetEnvironmentVariable("HOSHIKAGE_API_KEY", "User")) { "HOSHIKAGE_API_KEY is set" }
```

An application retains the environment it read when it started. This is why Codex must be restarted after adding or changing the variable.

### 4.6 Rotate and revoke

```bash
hoshikage token rotate codex-desktop
hoshikage token revoke codex-desktop
```

Rotation invalidates the old Token immediately. Read the replacement with `token list` and update the value that the upper application passes to Codex. When using a Windows user environment variable, update it and fully restart Codex. Revoke Tokens for retired clients or suspected disclosure.

Legacy digest-only Tokens continue to authenticate, but their plaintext cannot be reconstructed. When list shows `<unavailable: rotate required>`, rotate that name to migrate it to the new format.

### 4.7 Diagnosing 401 responses

1. Run `hoshikage token list` and confirm that the named Token exists.
2. Confirm that the Codex process `HOSHIKAGE_API_KEY` matches the Token shown by list.
3. Confirm that the upper application or Codex was restarted after updating the Token.
4. Confirm that Codex configuration uses `env_key = "HOSHIKAGE_API_KEY"`.
5. Never paste Token plaintext into logs, issues, or chat.

## 5. Connecting Codex

### 5.1 Where configuration belongs

Codex user configuration is separate from Hoshikage server configuration.

| Environment | Codex user configuration |
|---|---|
| Linux CLI | `~/.codex/config.toml` |
| macOS CLI | `~/.codex/config.toml` |
| Windows CLI and Codex app | `%USERPROFILE%\.codex\config.toml` |

Do not put Provider configuration in a workspace `.codex/config.toml`. For security, Codex ignores `model_provider` and `model_providers` in project configuration. `AGENTS.md` is also for working instructions, not server or model selection.

The Windows Codex app and a Codex CLI installed in WSL use different locations. The Windows app reads the Windows location; the WSL CLI normally reads WSL `~/.codex`.

### 5.2 Generating Provider configuration

On the Hoshikage server machine, generate configuration using its actual IP address.

```bash
hoshikage codex-config \
  --model unsloth-gemma4-12b-qat-thinking-off \
  --base-url http://192.168.1.50:3030/v1 \
  --authenticated
```

This command only prints configuration. It does not modify files on the Codex side. Its output has the following form.

```toml
model = "unsloth-gemma4-12b-qat-thinking-off"
model_provider = "hoshikage"
approval_policy = "on-request"
sandbox_mode = "workspace-write"
model_context_window = 65536
model_auto_compact_token_limit = 49152
tool_output_token_limit = 8192
model_reasoning_summary = "none"

[model_providers.hoshikage]
name = "Hoshikage"
base_url = "http://192.168.1.50:3030/v1"
wire_api = "responses"
env_key = "HOSHIKAGE_API_KEY"
request_max_retries = 1
stream_max_retries = 1

[sandbox_workspace_write]
network_access = true
```

Never put the Token plaintext in this TOML. `env_key` tells Codex which process environment variable contains the Token.

`network_access = true` is required for Skills such as Search, Fetch, and Recall to reach LAN and Internet services from inside the Codex sandbox. Writable paths remain restricted by `workspace-write`, and Hoshikage does not alter Codex approval or sandbox behavior.

### 5.3 Saving configuration by operating system

On Linux and macOS, save the generated content at:

```bash
mkdir -p ~/.codex
nano ~/.codex/config.toml
```

On Windows, open the location from PowerShell.

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\.codex"
notepad "$env:USERPROFILE\.codex\config.toml"
```

Save the generated TOML. If `config.toml` already exists, back it up first and merge the generated values without deleting other required settings.

Pass the Token to Codex using section 4.4 or 4.5, then start Codex and ask "Return exactly the word OK." in a new task. The setup is complete when Codex returns `OK` and the request appears in the Hoshikage server log.

### 5.4 Same-machine setup

No Token is required when Hoshikage and Codex run on the same machine and can connect through `127.0.0.1`. Save the following output to the operating-system-specific Codex user configuration from section 5.1.

```bash
hoshikage codex-config --model unsloth-gemma4-12b-qat-thinking-off
```

When the Windows app connects to Hoshikage inside WSL, first verify that `127.0.0.1` is reachable. If it is not, use the authenticated LAN procedure.

### 5.5 Selecting Hoshikage in the CLI

Use a named Profile when only selected Codex CLI runs should use Hoshikage. For a Profile named `hoshikage`, save the generated output to `~/.codex/hoshikage.config.toml` on Linux, macOS, or WSL, and `%USERPROFILE%\.codex\hoshikage.config.toml` on Windows.

Example for same-machine Hoshikage on Linux, macOS, or WSL:

```bash
mkdir -p ~/.codex
hoshikage codex-config \
  --model unsloth-gemma4-12b-qat-thinking-off \
  > ~/.codex/hoshikage.config.toml
codex --profile hoshikage "Return exactly the word OK."
```

On Windows, save the same TOML to `%USERPROFILE%\.codex\hoshikage.config.toml`, then run:

```powershell
codex --profile hoshikage "Return exactly the word OK."
```

Codex 0.134 and later do not use the old `[profiles.hoshikage]` form. Use the independent `hoshikage.config.toml` Profile file. For normal Windows Codex app use, prefer the user configuration in section 5.3.

### 5.6 Interactive and unattended operation

The default interactive configuration uses `approval_policy = "on-request"`. When you start `codex --profile hoshikage`, Codex's interactive UI lets you approve or reject operations that require permission, such as writes outside the workspace. Codex displays the approval request; Hoshikage does not.

`codex exec` is for non-interactive automation and does not wait for a user-facing approval UI. Create a separately named Profile only for a dedicated unattended environment.

```bash
mkdir -p ~/.codex
hoshikage codex-config \
  --model unsloth-gemma4-12b-qat-thinking-off \
  --mode unattended \
  > ~/.codex/hoshikage-unattended.config.toml
codex exec --profile hoshikage-unattended "Return exactly the word OK."
```

On Windows PowerShell, save and run it as follows:

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\.codex"
hoshikage codex-config `
  --model unsloth-gemma4-12b-qat-thinking-off `
  --mode unattended |
  Set-Content "$env:USERPROFILE\.codex\hoshikage-unattended.config.toml"
codex exec --profile hoshikage-unattended "Return exactly the word OK."
```

The unattended form generates `approval_policy = "never"`. This does not automatically permit operations that need approval: Codex runs only within the configured sandbox and returns an out-of-bounds operation to the model as a failure. Keep interactive and unattended configurations separate. Hoshikage does not control approvals or sandboxing; Codex does.

### 5.7 Configuration concepts

- **Provider**: Defines the Hoshikage API URL, Responses wire protocol, and authentication environment variable.
- **Profile**: Defines the model, Provider, approval policy, sandbox, and other Codex execution conditions.
- **Hoshikage server configuration**: Defines server binding, models, and the Token store. It is not Codex configuration.
- **Process environment variable**: Lets an upper application pass a Token to the Codex child process without putting it directly in configuration.
- **Model catalog**: A machine-readable list of Bundles and capabilities available to an upper application.
- **`AGENTS.md`**: Contains working policy and repository instructions. It does not select a server or model.

The application layer that starts Codex, such as Yatagarasu, owns model, Provider, and Token selection. Hoshikage never modifies upper-application configuration.

### 5.8 Model catalog and connection diagnostics

```bash
hoshikage codex-model-catalog --json
hoshikage doctor \
  --model unsloth-gemma4-12b-qat-thinking-off \
  --codex-base-url http://127.0.0.1:3030/v1
```

The catalog lists every Bundle with `codex_compatible`, context, Responses, streaming, and tools capabilities. It never exposes model file paths or Tokens.

When Tool Calling is `disabled`, Codex can use text responses only. To use file and shell tools in an Agent Loop, configure an appropriate `native` or `json` mode in the Bundle and verify it with `doctor`.

## 6. APIs and Status

### 6.1 Endpoints

- `GET /health`: unauthenticated liveness
- `GET /ready`: configuration and runtime readiness
- `GET /v1/models`: OpenAI-compatible model list
- `GET /v1/hoshikage/models`: Hoshikage model list with capabilities
- `GET /v1/status`: model loading state
- `POST /v1/chat/completions`: existing Chat Completions API
- `POST /v1/responses`: Responses API for Codex

### 6.2 Errors and request IDs

The Responses API returns OpenAI-compatible errors and an `x-request-id` header. Use the request ID, not request bodies or Tokens, when correlating logs or reporting a problem.

## 7. Logging and Debug Capture

Normal logs never contain prompts, Tool arguments, Tool results, or Tokens. They contain safe summaries such as request ID, model, timing, token counts, and terminal status.

Explicitly enable body capture only for a short diagnostic session.

```dotenv
HOSHIKAGE_DEBUG_CAPTURE=on
```

Captures are stored per request in `debug-capture` under the operating-system-specific Hoshikage configuration directory. Authorization, Token-named fields, and metadata are removed. Defaults are a 24-hour retention period and a 100 MiB directory limit; on Unix the directory is `0700` and files are `0600`. Startup emits a warning. Set the option back to `off` after diagnosis and treat capture files as confidential.

## 8. TLS and Networking

Hoshikage serves HTTP on the LAN. When traffic may be observed or crosses network segments, terminate TLS with a reverse proxy such as Caddy or nginx and bind Hoshikage only to loopback or a protected internal interface.

```text
Codex -> HTTPS reverse proxy -> HTTP Hoshikage
```

The proxy must forward the `Authorization` header and must not write request bodies to access logs. Never expose Hoshikage directly to the WAN.

## 9. Troubleshooting

### 9.1 Codex cannot connect

- Check the process with `curl http://HOST:3030/health`.
- Check runtime readiness with `/ready`.
- On a LAN, follow the 401 diagnostic procedure.
- Confirm that the Codex Provider `base_url` includes `/v1`.
- Run `doctor --codex-base-url` to check connectivity and model capabilities together.

### 9.2 Codex does not use tools

- Check `tools` with `hoshikage codex-model-catalog --json`.
- Confirm that Bundle `tool_calling.mode` is not `disabled`.
- Confirm that the parser matches the model chat template.
- Confirm that context is at least 16K.
- Hoshikage never executes tools. Also inspect the Codex Tool Registry, approval policy, and sandbox.

### 9.3 Selecting the display language

```bash
hoshikage --language ja doctor
hoshikage --language en doctor
```

Selection priority is `--language`, `HOSHIKAGE_LANG`, OS locale, then English fallback. Error codes, JSON fields, and diagnostic IDs are language-independent.
