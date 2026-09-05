"""Download the pinned public ONNX weights with bounded, verified range requests."""
import concurrent.futures
import hashlib
import urllib.request
from pathlib import Path

REVISION = "52398278842ec682c6f32300af41344b1c0b0bb2"
SHA256 = "51f1bd0addd6e859e42c2c8021a5e5461385bb676a649f4b269aa445449f2431"
SIZE = 66465124
URL = f"https://huggingface.co/qdrant/bge-small-en-v1.5-onnx-q/resolve/{REVISION}/model_optimized.onnx"
ASSETS = {
    "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    "config.json": "13582bcf2effc85b7bf3d3f5532e686bc1c9ce86bb009d10f0ec33cbe92299dd",
    "tokenizer_config.json": "0b29c7bfc889e53b36d9dd3e686dd4300f6525110eaa98c76a5dafceb2029f53",
    "special_tokens_map.json": "5d5b662e421ea9fac075174bb0688ee0d9431699900b90662acd44b2a350503a",
}


def fetch(start):
    end = min(start + 1024 * 1024, SIZE) - 1
    for attempt in range(3):
        try:
            request = urllib.request.Request(URL + f"?download=true&part={start}",
                headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read()
                assert response.headers.get("Content-Range") == f"bytes {start}-{end}/{SIZE}"
                assert len(data) == end - start + 1
                return data
        except Exception:
            if attempt == 2:
                raise


if __name__ == "__main__":
    target = Path(f"data/model_cache/models--qdrant--bge-small-en-v1.5-onnx-q/snapshots/{REVISION}/model_optimized.onnx")
    target.parent.mkdir(parents=True, exist_ok=True)
    for name, expected in ASSETS.items():
        path = target.parent / name
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            continue
        with urllib.request.urlopen(URL.rsplit("/", 1)[0] + "/" + name, timeout=30) as response:
            payload = response.read()
        assert hashlib.sha256(payload).hexdigest() == expected
        path.write_bytes(payload)
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == SHA256:
        print("Pinned model already verified")
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            parts = []
            for index, data in enumerate(pool.map(fetch, range(0, SIZE, 1024 * 1024)), 1):
                parts.append(data)
                print(f"Embedding weights: {index}/64 parts", flush=True)
        data = b"".join(parts)
        assert hashlib.sha256(data).hexdigest() == SHA256
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        print("Verified", SHA256)
