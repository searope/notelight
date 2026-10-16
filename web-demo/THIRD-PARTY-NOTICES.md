# Third-party notices for the browser demo

The demo page loads two things this repository does not contain. Neither is
committed; both are pinned and verified by SHA-256 before use.

## ONNX Runtime Web 1.30.0

The npm package `onnxruntime-web@1.30.0`, from which `scripts/vendor_ort.py`
extracts `ort.wasm.min.mjs`, `ort-wasm-simd-threaded.mjs` and
`ort-wasm-simd-threaded.wasm` into `web-demo/vendor/onnxruntime-web/`.
Source: <https://github.com/microsoft/onnxruntime> (tag `v1.30.0`).
The built files embed third-party code under their own licences, listed in
<https://github.com/microsoft/onnxruntime/blob/v1.30.0/ThirdPartyNotices.txt>.

```
MIT License

Copyright (c) Microsoft Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## BAAI/bge-small-en-v1.5

The embedding model, downloaded by the page from Hugging Face at revision
`5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` (`onnx/model.onnx`,
`tokenizer.json`) and kept in the browser's Cache API. MIT licence, as stated
on <https://huggingface.co/BAAI/bge-small-en-v1.5>. It is the same model and
revision the Python package uses.

`web-demo/tokenizer.js` reimplements the behaviour of Hugging Face
`tokenizers` (Apache-2.0) for this model's tokenizer; it contains no code
from that project.
