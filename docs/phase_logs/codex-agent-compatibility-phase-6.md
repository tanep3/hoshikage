# Codex Agent Compatibility Phase 6 作業ログ

## 2026-07-27

状態: Phase 6A Fix

### 目的

HoshikageをCodex CLIの推論Providerとして使用し、上位アプリケーションがローカルモデルを選択して既存Skillを安全に実行できることを実環境で証明する。

### 責務境界

- HoshikageはResponses API、Tool Call変換、モデルおよびruntime管理だけを担当する。
- Codex CLIはAgent Loop、Tool選択、Tool実行、承認、sandboxを担当する。
- Yatagarasuは最初の統合対象として、Codex Profile、モデル、Token、作業directoryを選択してCodexを起動する。
- View、Recall、Search、Fetch固有の実装や名称をHoshikageへ追加しない。

### Phase 6A実装順序

1. YatagarasuからCodexへ渡すHoshikage接続契約を固定する。
2. 対話用ProfileとToken受け渡しを実装する。
3. DoctorでProfile、Token、Hoshikage readinessを診断する。
4. mock Codexによる起動引数・環境変数の契約テストを行う。
5. View、Recall、Search、Fetchのread-only実機E2Eを行う。
6. Hoshikage全回帰と境界監査を行う。

### 開始時調査

- Hoshikageは対話用`on-request`と無人用`never`のCodex設定を生成できる。
- YatagarasuはCodex CLI、モデル、作業directoryを選択できる。
- YatagarasuはまだHoshikage用ProfileをCodexへ指定せず、TokenとProvider readinessも診断しない。
- Codex CLI 0.144.5は`--profile`で`$CODEX_HOME/<name>.config.toml`をlayerできる。
- YatagarasuのView、Recall、Search、FetchはCodex Skillとして既に分離されている。

### 実装

#### Hoshikage

- 対話用・無人用Codex Profileへ`[sandbox_workspace_write] network_access = true`を生成する。
- `workspace-write`の書込み境界は維持し、Search、Fetch、Recall等のnetwork-backed Skillだけを到達可能にした。
- 日本語・英語ユーザーマニュアルへsandbox networkの目的を同期して追記した。
- `/v1/models`はOpenAI互換`{object,data}`を維持し、Codex私有モデルカタログ形式へ変更しなかった。

#### Yatagarasu

- `YATAGARASU_CODEX_PROFILE`、`YATAGARASU_CODEX_MODEL`、`HOSHIKAGE_API_KEY`をCodex子processへ渡す。
- 対話利用の既定を`workspace-write`かつsandbox bypass無効とした。
- DoctorへProfile TOML、Provider、`wire_api`、Token有無、認証付き`/ready`、sandbox network、`uv`を追加した。
- `~/.local/bin`を非対話・systemd起動時の子process PATHへ追加した。
- `yatagarasu doctor`の既定診断先を実workspaceへ修正した。
- Recall Skillの存在しない裸の`recall`例をrepository内`recall.sh`の実行経路へ修正した。
- README、setup manual、`.env.example`へProfile、モデル、Token、sandbox、作業directoryの利用者手順を追加した。

### 本番反映

- Hoshikage server: `192.168.0.220:3030`
  - release binaryを`/home/tane/bin/hoshikage`へ配置した。
  - user systemd serviceを再起動し、active、binary hash一致、認証付き`/ready`成功を確認した。
  - LAN bindのため用途名`yatagarasu-phase6`のTokenを作成した。Token本文は本ログへ記録しない。
- Yatagarasu: `192.168.0.200:2202`
  - production treeへPhase 6A patchを適用した。
  - `$CODEX_HOME/yatagarasu-local.config.toml`とGit管理外`workspace/.env`へProvider、Bundle、Tokenを設定した。
  - `yatagarasu.service`の`YATAGARASU_CWD=/home/tane/tools/yatagarasu/workspace`を確認した。
  - 既存変更とSemanticMemory submodule状態は保持した。
- rollback用backup:
  - `/home/tane/bin/hoshikage.backup-20260727-phase6`
  - `/home/tane/backups/yatagarasu-phase6-20260727-1600.tar.gz`
  - 個別launcher、Doctor、Recall Skillの更新前backup

### 実機E2E

| 試験 | 結果 |
|---|---|
| Yatagarasu launcher -> Codex -> Hoshikage通常応答 | `OK` |
| Fetch | `FETCH_OK` |
| Recall | `RECALL_OK` |
| View | `VIEW_OK`、JPEG 320x240生成 |
| Search -> URL選択 -> Fetch | `SEARCH_FETCH_OK` |

全Skill試験はproduction serviceと同じYatagarasu workspace、`unsloth-gemma4-12b-qat-thinking-off`、Hoshikage Responses APIで実行した。Tool実行はCodexが担当し、HoshikageへSkill固有実装を追加していない。

### 回帰結果

- Hoshikage `cargo fmt --check`: PASS
- Hoshikage `cargo clippy --all-targets -- -D warnings`: PASS
- Hoshikage `cargo test --all-targets`: 226 PASS、1 ignored
  - unit: 212 PASS、1 ignored
  - contract fixtures: 12 PASS
  - manual parity: 2 PASS
- Yatagarasu local full pytest: 10 PASS
- Yatagarasu production full pytest: 10 PASS
- Yatagarasu Doctor production: 0 fail、1 warn

ignored 1件は既存のローカルllama.cpp実体依存probeであり、本Phaseで新規skipしていない。build scriptはllama.cpp headersがないためchecked-in FFI bindingsを使う旨を警告するが、Clippy code warningは0件である。

### 発生した失敗と対応

- 最初のYatagarasu patchはcorruptで`git apply`に失敗した。worktree変更がないことを確認し、構造単位で再作成した。
- sandbox内の`uv` cache、Docker API、localhost probeが権限制約で失敗した。必要な試験だけ明示承認下で再実行した。
- Yatagarasu統合テストはProfile、sandbox既定、Doctor診断を追加する各REDを確認後に実装した。
- 全pytest初回は`PYTHONPATH`不足で既存`intent_router`を収集できなかった。`PYTHONPATH=.`を明示して10件PASSを確認した。
- production SSHではCodexとNodeが非対話PATHになく2回失敗した。nvm内実体を確認し、Yatagarasu launcherのcommand解決経路で成功した。
- Fetch直試験は`uv`が非対話PATHになくexit 127となった。launcherとDoctorが`~/.local/bin`を継承するようTDD修正した。
- `yatagarasu doctor`が呼出元directoryをworkspaceとして誤診断した。Doctor subcommandの既定をproject workspaceへTDD修正した。
- Recall Skill文書は存在しない裸の`recall`コマンドを例示していた。repository scriptの位置引数形式へTDD修正した。
- RecallはCodex sandboxのnetwork既定無効により`localhost:6001`へcurl exit 7となった。生成Profileへ`network_access = true`を追加し、Doctor診断対象にした。
- 手動E2Eをproject rootから実行した際はSkill auto-discovery対象外となった。production serviceと同じworkspaceで再実行し成功した。
- Recall詳細診断時、Codex CLI stderrがTool Result本文を表示した。Hoshikage logは本文を保持せず、Yatagarasu launcherも成功時stderrを利用者へ表示しない。運用診断でCodex詳細logを共有する場合はTool Resultを秘密情報として扱う。
- release buildを並行実行した2回は完了前のbuild lockを誤認し、旧binary hashが残った。単一jobで完了コードと生成Profile本文を確認してから再配置した。

### 残存事項

- Codex CLI 0.145.0はcustom Providerの`GET /models`へOpenAI公開仕様とは異なる私有`{models:[ModelInfo...]}`を要求し、Hoshikageの標準`{object,data}`にdecode errorを記録する。明示モデルではAgent Loopは継続し、全E2Eは成功した。
- `remote_models = false`はCodex 0.145系で削除済み互換フラグとなり、このrefreshを停止しない。試作設定は撤回した。
- Hoshikageの標準`/v1/models`をCodex私有schemaへ変更するとOpenAI互換性と責務境界を壊すため行わない。
- 将来の解消候補は、HoshikageのCodex用モデルカタログを`model_catalog_json`として安全に配置する上位アプリケーション契約である。ただしCodexの完全な`ModelInfo`はbase instructions等のAgent metadataを含むため、Hoshikageが安易に所有しない。Phase 6B前の独立検討事項とする。
- Doctorの1 warnはcustom model IDがCodex標準`models_cache.json`にないことによる。明示したcontext、compact、tool output制限で実行は継続する。
- `codex exec`はProfileが`approval_policy = "on-request"`でも非対話実行表示が`approval: never`となる。副作用系SkillはPhase 6Bで承認境界を別途実機検証する。

### Phase 6A Fix後の本番障害

- 症状: 音声wakeと文字起こしは成功するが、Yatagarasuが返答しない。
- Token: `workspace/.env`に設定済みで、Hoshikageへ認証付きrequestが到達していた。
- 原因: production `yatagarasu.service`が`.env`更新前から継続稼働し、listend process内に旧`YATAGARASU_MODEL=gpt-5.4-mini`とsandbox bypass設定を保持していた。子launcherはprocess環境を`.env`より優先するため、更新済みBundleとsandboxが選択されなかった。
- 失敗結果: Codexが旧モデルmetadataに基づく`custom` Toolを送信し、Hoshikageが`unsupported_tool_type`として正しく拒否した。
- 復旧: `systemctl --user restart yatagarasu.service`で`.env`を再読込した。
- 復旧確認: service active、SBERT Router ready、PTZ worker ready、audio read loop、heartbeat、同一workspaceから`VOICE_PIPELINE_OK`を確認した。
- 恒久運用: production `.env`を変更した場合は、Doctorだけでなく長時間稼働中のYatagarasu音声serviceを再起動する。

### 完了条件

- Yatagarasuがアプリケーション設定からHoshikage Profileと任意の対応モデルを選択できる。
- LAN利用時、Yatagarasuがserver側Token fileを参照せず、設定されたTokenをCodex子processへ渡す。
- 対話用実行で`on-request`と`workspace-write`の境界を維持する。
- View、Recall、Search、Fetchが逐次Agent Loopとして完走する。
- Hoshikage境界にYatagarasuまたはSkill固有実装が存在しない。
- 失敗、skip、実機依存の未実施項目を本ログへ明記する。

### Phase 6A Fix後のAgent context是正

- 16K BundleではCodex初期指示とSkill記述だけで約9K tokensを消費し、12Kの自動圧縮閾値へ早期到達することを確認した。
- Tool結果の4KB制限はHoshikageへ再投入する本文の制御であり、CodexがTool実行直後に見る生出力やモデルcontextの代替ではなかった。
- 標準`thinking-off`と`thinking-on` Bundleを64Kへ拡張し、Tool結果のhead-tail保持を4KBから16KBへ拡張した。
- 64K/F16 KV/MTP有効は短いResponses推論に成功したが、RTX 4070 SUPER 12GBで空きVRAMが424MiBしかなく、本番設定として不採用とした。
- 64K/Q8 KV/MTP有効は空き771MiB、比較用の64K/Q8 KV/MTP無効は空き1208MiBだった。ただしMTPはHoshikage採用価値を構成する必須速度要件であるため、無効構成は不採用とした。標準Agent Bundleは64K/Q8 KV/MTP有効で検証を継続する。
- managed llama-serverのK/V cache型を型付き環境設定からmain/draft双方へ渡す実装をTDDで追加した。不正値は起動時に拒否し、未設定時は既存挙動を維持する。
- Yatagarasu Searchはstdoutへ上位10件の構造化JSONだけを返し、ログをstderrへ分離する契約へ変更した。各結果はtitle、URL、500 bytes以下のsnippet、published date、engineを保持し、詳細本文はFetchへ分離する。
- Hoshikage releaseへ64K/Q8 KV/MTP有効を反映し、実起動引数にmain/draft双方の`--cache-type-*=q8_0`と`--spec-type draft-mtp`が含まれることを確認した。
- Yatagarasu本番Profileを`model_context_window=65536`、`model_auto_compact_token_limit=49152`、`tool_output_token_limit=8192`へ同期した。
- 最初に障害となった自然文「明日の埼玉県入間市の天気を教えて。」をthinking-off/onの双方で再実行し、Search 1回、Fetch 1回、最終回答で終了コード0となった。
- 両セッションともSearchは`returned_results=10`、Codex compactionなし、同一Tool反復なしだった。Hoshikage側は各3 requestをすべて`completed`で終え、最終requestの入力はthinking-off 19235 tokens、thinking-on 19526 tokensだった。
- 64K/Q8 KV/MTP有効の試験時VRAM空きはWindows側利用量に応じて約466-771MiBで変動した。MTPを維持した上でOOM監視が必要な運用リスクとして残す。

#### 是正中に発生した失敗

- 64K設定後の最初のヘルスチェックはsystemd再起動直後に待機せず実行し、connection refusedとなった。サービスactiveを確認し、待機付き再試験で`{"status":"ok"}`を確認した。
- standalone llama-cliのメタデータ確認は対話promptが連続出力されtimeoutした。モデル公称値の推測には使わず、managed llama-serverの実ロードとVRAM実測を判断根拠にした。
- Searchの10件契約REDは、旧scriptが人向けログをstdoutへ混在させていたためJSON decode errorで失敗した。stdout/stderr契約を分離後にPASSした。
- Yatagarasu全体へ引数なしでpytestを実行した試験は、管理対象外のSemanticMemory submoduleと`python/` import pathを誤収集して2件のcollection errorとなった。正式対象を`PYTHONPATH=python ... pytest python/tests`と明示し、13件PASSを確認した。
- Yatagarasu本番で最初に`uv run pytest`を実行した試験は、production root環境にpytest executableがなく起動失敗した。既存`python/.venv/bin/pytest`を使って同じ13件を再実行しPASSした。
