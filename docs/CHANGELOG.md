# 星影 - 更新履歴

## 2026-01-18

### ドキュメントの更新
1. **README.mdの移動と再作成**
   - Python用のREADME.mdを`legacy_python/`に移動
   - Rust版用のREADME.mdを作成
   - 動的リンクの仕組みを説明

2. **user-manual.mdの更新**
   - インストール方法を`cargo install --path .`に変更
   - モデル管理方法を`~/.config/hoshikage/model_map.json`ベースに変更
   - ライブラリのトラブルシューティングを追加
   - トラブルシューティング（コマンド、ポート）のセクションを削除
   - 古いセクション（ログ、アンインストール、パフォーマンスチューニング）を削除

3. **ドキュメントの整理**
   - `docs/LIBRARY_FILES.md` - ライブラリファイルの配置場所と種類を説明
   - `docs/CHANGELOG.md` - 変更履歴の更新

### 動的リンクについて

**現在の構成:**
- llama.cppの静的ライブラリ（.aファイル）はバイナリに埋め込まれています
- システムCUDAライブラリ（.soファイル）は実行時に動的リンクされます
- 環境変数`LD_LIBRARY_PATH`でCUDAライブラリの検索パスを設定

**推奨される使用方法:**

```bash
# システムCUDAライブラリを使用する場合（推奨）
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
./target/release/hoshikage
```

**カスタムCUDAライブラリを使用する場合:**

```bash
# システムCUDAライブラリからコピー
mkdir -p ~/.config/hoshikage/lib
cp /usr/local/cuda/targets/x86_64-linux/lib/libcuda.so ~/.config/hoshikage/lib/
cp /usr/local/cuda/targets/x86_64-linux/lib/libcublas.so ~/.config/hoshikage/lib/
cp /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so ~/.config/hoshikage/lib/

# 環境変数を設定
export LD_LIBRARY_PATH=~/.config/hoshikage/lib:$LD_LIBRARY_PATH
./target/release/hoshikage
```

---

**作成日:** 2026-01-18

