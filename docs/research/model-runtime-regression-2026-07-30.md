# Model Runtime Regression Investigation (2026-07-30)

## 結論

1. Responses API未指定時の`max_output_tokens=1024`補完、Chat Completionsの
   `1024/2096`上限、FFI生成loopの`4096`上限はHoshikage側の人工的な制約だった。
2. LFM2.5-Q8の複雑なRust課題の誤答はmanaged runtimeと旧FFIの双方で再現した。
   managed移行による品質regressionではない。旧FFIはEOS後も生成を続け、結果は悪化した。
3. ELYZA Diffusionは通常のautoregressive生成ではなく反復denoiseを必要とする。
   managed llama-serverへ送る経路は不適切で、既存Diffusion FFI経路へ戻す必要がある。

## 修正

- 出力上限の未指定を`Option<u32>`としてResponses wireからinference contractまで保持する。
- managed llama-serverには未指定の`max_tokens`を送らない。
- FFIは入力token数を実測し、Bundle contextの残容量を生成上限とする。
- `ModelConfig.generation`へ`autoregressive`と`diffusion`を追加する。
- global runtimeが`llama-server-managed`でも、Diffusion BundleだけFFIへrouteする。
- FFI Responses streamingはbuffered completionを正規SSE event列へ変換する。

## 一次情報

- Liquid AI公式Model Card:
  https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct
  - context length: 32,768
  - 推奨: temperature 0.1、top_k 50、repetition_penalty 1.05
  - agentic tasks、data extraction、RAG向け。programmingは非推奨。
- ELYZA公式Model Card:
  https://huggingface.co/elyza/ELYZA-Diffusion-Instruct-1.0-Dream-7B
  - DDMLMとしてall-MASK列から反復denoiseする。
  - 公式例は`diffusion_generate`を使用する。
- llama.cpp公式server文書:
  https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
  - OpenAI互換chat/completion、multimodal、speculative decodingは記載される。
  - ELYZAが必要とするDiffusion生成APIは提供されていない。

## 実モデル確認

- ELYZA Diffusion non-stream Responses: HTTP 200、32 output tokensで完了。
- ELYZA Diffusion stream Responses:
  `response.created`から`response.completed`まで完走。
- LFM2.5-Q8基本Rust関数: 生成後にRust 2021でcompile成功。
- LFM2.5 Instruct/JP Bundle: 公式contextに合わせ`n_ctx=32768`へ更新。
- 登録10モデルを維持。

複雑なRust課題をLFM2.5のruntime受け入れ条件にはしない。runtime回帰試験は、同一promptを
新旧経路へ通した出力、停止条件、token usage、HTTP/SSE完了を分離して比較する。
