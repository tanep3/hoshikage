# Codex Agent Compatibility Phase 6 作業ログ

## 2026-07-27

状態: Phase 6 Fix

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

- Codex CLI 0.145.0はcustom Providerの`GET /models`へOpenAI公開仕様とは異なる私有`{models:[ModelInfo...]}`を要求する。Phase 6Bで標準`{object,data}`を維持したままCodex用`models`を併記する後方互換方式へ改訂した。
- `remote_models = false`はCodex 0.145系で削除済み互換フラグとなり、このrefreshを停止しない。試作設定は撤回した。
- Codex私有metadataは相互運用境界として必要最小限だけ生成し、Bundleのpathや秘密情報は含めない。Agent Loopの基礎指示、context、入力modality、shell利用可否、逐次Tool Call方針を明示する。
- `custom` Toolは本Phase対象外のため、`apply_patch_tool_type`を広告しない。通常のFunction Toolとshell ToolだけをCodexへ公開する。
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

## Phase 6B: 副作用系

状態: Phase 6B Fix

### 開始判断

- 利用者がPhase 6A Fixを承認したため、Phase 6Bを開始した。
- Yatagarasuの読取系統合はPhase 6Aとして完了している。ただしPhase 6BのMemorize実機試験で上位SkillとSemanticMemoryに未検出障害が見つかったため、書込系まで含む統合完了判定は保留した。
- Codex公式manualで、対話実行は`approval_policy = "on-request"`、無人実行は承認UIを持たない`codex exec`であることを再確認した。

### Hoshikage実装

- `function_call_output.status`の`completed`、`success`、`failed`、`error`、`rejected`、`cancelled`を型付き`ToolOutcome`へ変換する契約テストを追加した。
- `status`が数値等の非文字列でも従来は未指定と同じSuccessへ変換されていた。REDで再現し、`invalid_request`として拒否するよう修正した。
- Success、Failure、Rejected、Cancelledをllama-serverのTool Resultへ意味を保持して再投入する回帰テストを追加した。
- HoshikageはTool結果本文から成否を推測せず、wire上の明示statusだけを`ToolOutcome`へ変換する責務境界を維持した。

### Codex承認境界

- 対話Profileは`codex --profile <name>`で起動し、Codex UIが承認を表示する。
- 無人Profileは別名で生成し、`codex exec --profile <unattended-name>`から使用する。
- `approval_policy = "never"`はsandbox外操作の自動許可ではない。Codexは設定済みsandbox内だけで実行し、境界外操作を失敗としてAgent Loopへ返す。
- 日本語・英語manualのコマンド、保存先、Windows PowerShell例を同期して修正した。

### Yatagarasu受け入れ修正

- Memorize Skill文書が存在しない裸の`memorize`コマンドを案内し、script自身もhelp例の引用符不正でshell解析に失敗していた。
- Skill文書をrepository内script pathへ統一し、shell構文テストを追加した。
- curl HTTP失敗を無言のexit codeとして返していたため、timeout、HTTP error検出、秘密本文を出さない明示エラーを追加した。
- Skillへ「終了コード0、`status: saved`、保存IDの3点が揃った場合だけ成功」と定義した。
- 会話自動保存用`bin/memorize.sh`の重複実装137行を廃止し、同じAgentSkill scriptへ委譲した。
- Codex child processが`.env`を既に読み込んでいる場合、env readerが`set -e`で無言終了する欠陥を、実`.env`を再現するテストで修正した。
- Yatagarasu mainへ次をcommitし、NAS、GitHub、本番へ順次同期した。
  - `60705c7 fix: restore Memorize skill execution`
  - `63c1ce9 fix: make Memorize failures explicit`
  - `bc093be fix: require verified Memorize execution`
  - `a226e6e fix: preserve preloaded Memorize environment`
  - `239ea96 fix: deduplicate repeated Memorize side effects`
- 同一保存要求がTool待機中に再実行されても副作用を重複させないよう、payload hash、排他lock、60秒の保存ID cacheによる冪等化を追加した。cacheは本文を保持しない。
- 本番`workspace/.env.example`のWake Word差分は`60705c7`へ正式化し、本番の親repository差分を解消した。
- 本番で残る`external/SemanticMemory`のmodified表示は、submodule内4ファイルの既存未commit patchである。親repositoryへ混入させず保護している。

### SemanticMemory運用修正

- 要約付き`/api/save`が500となる原因は、設定モデル`gemini-3-flash-preview:cloud`へのOllama generateが410を返すことだった。
- 本番に存在し、generate成功を確認したローカル`qwen3.5:0.8b`へ要約モデル設定を変更した。
- 要約付きprobeは保存ID 282を返し、試験後に削除した。

### 実機E2E

| 試験 | 結果 |
|---|---|
| Hoshikage Tool outcome全種のwire変換 | PASS |
| Tool outcome全種のllama-server再投入 | PASS |
| Yatagarasu Memorize script構文・HTTP失敗 | PASS |
| SemanticMemory要約付き直接保存 | PASS、ID 282を削除 |
| thinking-off自然文、会話自動保存なし | FAIL、Tool Callなしで偽成功回答 |
| thinking-on自然文、会話自動保存なし（env reader修正前） | Tool Call生成、script無言exit 1、最終回答は失敗 |
| thinking-on自然文、env reader修正後 | Tool Call待機中に同一保存を再実行し、ID 284、285を重複生成。両方削除 |
| Memorize冪等化後のthinking-on自然文 | FAIL、Tool Callなし、DB保存0件 |
| thinking-on明示Skill実行、会話自動保存なし | PASS、保存ID 286をDBで1件だけ確認し削除後0件 |
| unattended workspace外書込拒否 | PASS、workspace内実行証跡を作成後、同一Tool Callの外部file作成だけが失敗。試験証跡を削除 |
| Codexモデルカタログ読込 | PASS、`/v1/models` decode警告とfallback metadata警告を解消 |
| モデルカタログ経由Function Tool | PASS、`exec_command`出力`MODEL_CATALOG_TOOL_OK`、最終応答`MODEL_CATALOG_OK` |

thinking-offとthinking-onの双方で、自然な保存依頼からToolを選択しない試行がある。HoshikageのTool変換と明示Skill経路は完走しているため、Provider契約とは分離したモデル選択品質として残る。自動会話保存を有効にした試験結果をMemorize Skill成功と誤認しないよう、最終判定では`YATAGARASU_MEMORY_ENABLED=false`を必須とする。

workspace外書込拒否の初回はfileが作成されなかったがTool実行eventもなく、モデルが実行せず失敗と回答した可能性を排除できなかったため不合格とした。再試験ではworkspace内fileを先に作る同一shell commandを使用し、Tool実行を証明した上で外部fileだけが存在しないことを確認した。

Codexモデルカタログの初回実装は`apply_patch_tool_type=freeform`を過大広告し、Codexが未対応の`custom` Toolを送って`unsupported_tool_type`となった。広告をFunction/shell能力へ限定し、同じ本番経路で再試験して完走した。

### Phase 6B回帰

- Hoshikage `cargo fmt --check`: PASS
- Hoshikage `cargo clippy --all-targets -- -D warnings`: PASS
- Hoshikage `cargo test --all-targets`: 234 PASS、1 ignored
  - unit: 220 PASS、1 ignored
  - contract fixtures: 12 PASS
  - manual parity: 2 PASS
- Yatagarasu `PYTHONPATH=python python/.venv/bin/pytest python/tests`: 16 PASS

既存ignored 1件はローカルllama.cpp実体依存probeであり、Phase 6Bで新規skipしていない。

### Phase 6B残存判断

1. HoshikageのResponses、Function Tool、Tool Result、承認・sandbox境界は実機で成立した。
2. 明示されたMemorize Skillはthinking-onで実保存まで完走し、重複副作用も防止した。
3. 自然文だけから副作用Toolを選ぶ確率はthinking-off、thinking-onとも保証できない。明示Skill経路では成功しており毎回失敗するものではないため、Gemma 4のモデル能力制約として利用者が受容した。
4. 2026-07-27、利用者承認によりPhase 6B Fix。これをもってCodex Agent Compatibility初期実装のPhase 0からPhase 6を完了とする。
