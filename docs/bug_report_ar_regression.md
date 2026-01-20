# Bug Report: AR Model Generation Regression

## Problem Summary
The Autoregressive (AR) model generation is "running away" (producing infinite output or ignoring stop conditions).
Investigation reveals a critical regression in `src/inference/llama_wrapper.rs`.

## Cause Analysis
In the `generate_with_callback` method, the logic to **check for the EOS (End of Sentence) token is completely missing**.
After sampling a new token, the code immediately proceeds to decode it and append it to the result, without checking if the model intentionally signaled the end of generation.

**Missing Logic**:
```rust
let token_eos = llama_token_eos(model); // Missing symbol load & call
// ...
let token = sample_token(...)?;
if token == token_eos { // Missing check
    break;
}
```

## Solution Plan
We need to patch `src/inference/llama_wrapper.rs` to verify the EOS token.

### Tasks for AI System
1.  **Load Symbol**: In `generate_with_callback`, dynamically load `llama_token_eos` symbol from the library.
2.  **Get EOS Token**: Retrieve the EOS token ID for the current model.
3.  **Add Check**: Inside the generation loop, immediately after `sample_token`, add a check: `if token == eos_token { break; }`.

This will restore the correct termination behavior for standard AR models.
