import json
import numpy as np
import zarr

# Load quantization params
with open('/Users/kdoerkse/Documents/ASU/JPL/olmo/embeddings/healpix_0_quantization_params.json') as f:
    params = json.load(f)

data8 = zarr.open('/Users/kdoerkse/Documents/ASU/JPL/olmo/embeddings/healpix_0_embeddings_int8.zarr', mode='r')
embeddings_int8 = data8[:]

data32 = zarr.open('/Users/kdoerkse/Documents/ASU/JPL/olmo/embeddings/healpix_0_embeddings_float32.zarr', mode='r')
embeddings = data32[:]

# Dequantize int8 back to float32
embeddings_dequantized = np.zeros_like(embeddings_int8, dtype=np.float32)
for i in range(len(params['scales'])):
    embeddings_dequantized[i] = (embeddings_int8[i].astype(np.float32) + 128) * params['scales'][i] + params['offsets'][i]

# Compare
print(f"Original float32 range:  [{embeddings.min():.4f}, {embeddings.max():.4f}]")
print(f"Dequantized range:       [{embeddings_dequantized.min():.4f}, {embeddings_dequantized.max():.4f}]")

# Error metrics
mae = np.mean(np.abs(embeddings - embeddings_dequantized))
relative_error = mae / np.mean(np.abs(embeddings))

print(f"\nError metrics:")
print(f"  Mean Absolute Error: {mae:.6f}")
print(f"  Relative Error:      {relative_error:.2%}")

# Cosine similarity (embedding quality)
cos_sim = np.mean([
    np.dot(embeddings[i].flatten(), embeddings_dequantized[i].flatten()) /
    (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings_dequantized[i]) + 1e-8)
    for i in range(embeddings.shape[0])
])
print(f"  Cosine Similarity:   {cos_sim:.6f} (>0.99 = excellent)")