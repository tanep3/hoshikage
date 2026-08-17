# Hoshikage Documentation

Hoshikageの正規ドキュメントと作業記録への入口です。

## Codex Agent Compatibility

読む順序:

1. [要件定義書](codex-agent-compatibility-requirements.md)
2. [システム設計書](codex-agent-compatibility-system-design.md)
3. [互換性マトリクス](codex-agent-compatibility-matrix.md)
4. [Phase 0契約観測報告](research/codex-agent-compatibility-phase-0.md)
5. [Phase 0作業ログ](phase_logs/codex-agent-compatibility-phase-0.md)
6. [Phase 1作業ログ](phase_logs/codex-agent-compatibility-phase-1.md)
7. [Phase 2作業ログ](phase_logs/codex-agent-compatibility-phase-2.md)
8. [Phase 3作業ログ](phase_logs/codex-agent-compatibility-phase-3.md)
9. [Phase 4作業ログ](phase_logs/codex-agent-compatibility-phase-4.md)
10. [Phase 5作業ログ](phase_logs/codex-agent-compatibility-phase-5.md)
11. [Phase 6作業ログ](phase_logs/codex-agent-compatibility-phase-6.md)
12. [Phase 7 Vision作業ログ](phase_logs/codex-agent-compatibility-phase-7.md)

要件定義書とシステム設計書を正規仕様とする。`research/`は再利用可能な調査結果、
`phase_logs/`は実装・テスト・失敗を含む時系列記録である。

## Existing Hoshikage

- [プロジェクト概要](../README.md)
- [API仕様](api-spec.md)
- [ユーザーマニュアル（日本語）](user-manual.md)
- [User Manual (English)](user-manual.en.md)
- [非機能要件詳細](nfr-details.md)
- [Model Runtime Revision 要件](model-runtime-revision-requirements.md)
- [Model Runtime Revision 設計](model-runtime-revision-system-design.md)
- [Model Runtime Regression調査 (2026-07-30)](research/model-runtime-regression-2026-07-30.md)

実装済みの主要な接続確認は、Hoshikageの`/health`、`/v1/models`、`/v1/responses`、
およびCodex Hoshikage Proxy経由のモデル一覧・Responses疎通で行う。作業ログと受け入れ条件は、
各要件書・設計書・Phaseログを正とする。
