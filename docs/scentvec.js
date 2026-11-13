/*
 * scentvec.js: the scentvec perfume encoder, ported to plain JavaScript.
 *
 * No dependencies.  Loads model.json + model.bin (written by
 * scripts/export_web_model.py) and reproduces the PyTorch forward pass:
 *
 *   notes with pyramid positions
 *     -> one vector per note (+ position), layer norm
 *     -> one pre-norm self-attention layer (4 heads) + feed-forward
 *     -> layer norm, mean pooling and attention pooling
 *     -> projection, unit length                       => 128 numbers
 *
 * Only notes that have their own id in the model (1,053 of them) are
 * supported here.  The Python package also embeds unseen notes through their
 * words and an odor-chemistry profile.
 *
 *   const model = await ScentVec.load("./");
 *   const a = model.embed({ top: ["bergamot"], middle: ["rose"], base: ["musk"] });
 *   model.describe(a.vector).accords[0];   // { label: "Floral", score: 0.87 }
 */
(function (root) {
  "use strict";

  const POSITION_ORDER = ["top", "middle", "base", "notes"];

  function decodeFloat16(buffer) {
    const view = new DataView(buffer);
    const count = buffer.byteLength >> 1;
    const output = new Float32Array(count);
    for (let index = 0; index < count; index++) {
      const bits = view.getUint16(index * 2, true);
      const exponent = (bits >> 10) & 0x1f;
      const fraction = bits & 0x3ff;
      let value;
      if (exponent === 0) {
        value = fraction * Math.pow(2, -24);
      } else if (exponent === 31) {
        value = fraction ? NaN : Infinity;
      } else {
        value = (1 + fraction / 1024) * Math.pow(2, exponent - 15);
      }
      output[index] = bits & 0x8000 ? -value : value;
    }
    return output;
  }

  // Abramowitz & Stegun 7.1.26, absolute error below 1.5e-7.
  function erf(x) {
    const sign = x < 0 ? -1 : 1;
    const ax = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * ax);
    const polynomial =
      ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
    return sign * (1 - polynomial * Math.exp(-ax * ax));
  }

  function geluInPlace(values) {
    for (let index = 0; index < values.length; index++) {
      const x = values[index];
      values[index] = 0.5 * x * (1 + erf(x * Math.SQRT1_2));
    }
    return values;
  }

  // rows x inDim  ->  rows x outDim, with weight stored as [outDim, inDim].
  function linear(input, rows, inDim, weight, bias, outDim) {
    const output = new Float32Array(rows * outDim);
    for (let row = 0; row < rows; row++) {
      const inputOffset = row * inDim;
      const outputOffset = row * outDim;
      for (let out = 0; out < outDim; out++) {
        const weightOffset = out * inDim;
        let sum = bias[out];
        for (let k = 0; k < inDim; k++) {
          sum += input[inputOffset + k] * weight[weightOffset + k];
        }
        output[outputOffset + out] = sum;
      }
    }
    return output;
  }

  function layerNorm(input, rows, dim, weight, bias, eps) {
    const output = new Float32Array(rows * dim);
    for (let row = 0; row < rows; row++) {
      const offset = row * dim;
      let mean = 0;
      for (let k = 0; k < dim; k++) {
        mean += input[offset + k];
      }
      mean /= dim;
      let variance = 0;
      for (let k = 0; k < dim; k++) {
        const centred = input[offset + k] - mean;
        variance += centred * centred;
      }
      const scale = 1 / Math.sqrt(variance / dim + eps);
      for (let k = 0; k < dim; k++) {
        output[offset + k] = (input[offset + k] - mean) * scale * weight[k] + bias[k];
      }
    }
    return output;
  }

  function softmaxInPlace(values) {
    let maximum = -Infinity;
    for (let index = 0; index < values.length; index++) {
      maximum = Math.max(maximum, values[index]);
    }
    let total = 0;
    for (let index = 0; index < values.length; index++) {
      values[index] = Math.exp(values[index] - maximum);
      total += values[index];
    }
    for (let index = 0; index < values.length; index++) {
      values[index] /= total;
    }
    return values;
  }

  function cosine(first, second) {
    let dot = 0;
    let firstNorm = 0;
    let secondNorm = 0;
    for (let index = 0; index < first.length; index++) {
      dot += first[index] * second[index];
      firstNorm += first[index] * first[index];
      secondNorm += second[index] * second[index];
    }
    const denominator = Math.sqrt(firstNorm) * Math.sqrt(secondNorm);
    return denominator > 0 ? dot / denominator : 0;
  }

  class Model {
    constructor(manifest, buffer) {
      if (manifest.format !== 1 || manifest.dtype !== "float16-le") {
        throw new Error("Unsupported model format.");
      }
      if (buffer.byteLength !== manifest.binary_bytes) {
        throw new Error(
          "model.bin is " + buffer.byteLength + " bytes, expected " + manifest.binary_bytes + "."
        );
      }
      const values = decodeFloat16(buffer);
      this.manifest = manifest;
      this.dim = manifest.hidden_dim;
      this.embedDim = manifest.embed_dim;
      this.heads = manifest.n_heads;
      this.eps = manifest.layer_norm_eps;
      this.maxNotes = manifest.max_notes;
      this.notes = manifest.notes;
      this.noteIndex = new Map(manifest.notes.map((note, index) => [note, index]));
      this.tensors = {};
      for (const [name, entry] of Object.entries(manifest.tensors)) {
        this.tensors[name] = values.subarray(entry.offset, entry.offset + entry.length);
      }
    }

    has(note) {
      return this.noteIndex.has(note);
    }

    /*
     * pyramid: { top: [...], middle: [...], base: [...], notes: [...] }
     * ("notes" = position unknown).  A note listed twice keeps its first place.
     * Returns { vector, used: [{note, position}], unknown: [note, ...] }.
     * With no usable notes the vector is all zeros, as in the Python package.
     */
    embed(pyramid) {
      const used = [];
      const unknown = [];
      const seen = new Set();
      POSITION_ORDER.forEach((key, positionCode) => {
        for (const raw of pyramid[key] || []) {
          const note = String(raw).trim().toLowerCase();
          if (!note || seen.has(note)) {
            continue;
          }
          seen.add(note);
          if (!this.noteIndex.has(note)) {
            unknown.push(note);
          } else if (used.length < this.maxNotes) {
            used.push({ note: note, position: positionCode, index: this.noteIndex.get(note) });
          }
        }
      });
      const vector = used.length ? this._forward(used) : new Float32Array(this.embedDim);
      return {
        vector: vector,
        used: used.map((entry) => ({ note: entry.note, position: POSITION_ORDER[entry.position] })),
        unknown: unknown,
      };
    }

    _forward(entries) {
      const t = this.tensors;
      const d = this.dim;
      const n = entries.length;

      let x = new Float32Array(n * d);
      entries.forEach((entry, row) => {
        const noteOffset = entry.index * d;
        const positionOffset = entry.position * d;
        for (let k = 0; k < d; k++) {
          x[row * d + k] = t.note_vectors[noteOffset + k] + t.position[positionOffset + k];
        }
      });
      x = layerNorm(x, n, d, t.input_norm_weight, t.input_norm_bias, this.eps);

      // Self-attention block (pre-norm): x = x + attention(norm1(x))
      const normed = layerNorm(x, n, d, t.norm1_weight, t.norm1_bias, this.eps);
      const qkv = linear(normed, n, d, t.attn_in_weight, t.attn_in_bias, 3 * d);
      const headDim = d / this.heads;
      const scale = 1 / Math.sqrt(headDim);
      const context = new Float32Array(n * d);
      const weights = new Float32Array(n);
      for (let head = 0; head < this.heads; head++) {
        const queryStart = head * headDim;
        const keyStart = d + head * headDim;
        const valueStart = 2 * d + head * headDim;
        for (let i = 0; i < n; i++) {
          for (let j = 0; j < n; j++) {
            let score = 0;
            for (let k = 0; k < headDim; k++) {
              score += qkv[i * 3 * d + queryStart + k] * qkv[j * 3 * d + keyStart + k];
            }
            weights[j] = score * scale;
          }
          softmaxInPlace(weights);
          for (let j = 0; j < n; j++) {
            const weight = weights[j];
            for (let k = 0; k < headDim; k++) {
              context[i * d + head * headDim + k] += weight * qkv[j * 3 * d + valueStart + k];
            }
          }
        }
      }
      const attended = linear(context, n, d, t.attn_out_weight, t.attn_out_bias, d);
      for (let index = 0; index < x.length; index++) {
        x[index] += attended[index];
      }

      // Feed-forward block: x = x + ff(norm2(x))
      const normed2 = layerNorm(x, n, d, t.norm2_weight, t.norm2_bias, this.eps);
      const inner = geluInPlace(linear(normed2, n, d, t.ff1_weight, t.ff1_bias, 2 * d));
      const fed = linear(inner, n, 2 * d, t.ff2_weight, t.ff2_bias, d);
      for (let index = 0; index < x.length; index++) {
        x[index] += fed[index];
      }
      x = layerNorm(x, n, d, t.output_norm_weight, t.output_norm_bias, this.eps);

      // Pool the notes two ways: plain mean, and a learned attention weighting.
      const pooled = new Float32Array(2 * d);
      const poolWeights = softmaxInPlace(
        linear(x, n, d, t.pool_score_weight, t.pool_score_bias, 1)
      );
      for (let row = 0; row < n; row++) {
        for (let k = 0; k < d; k++) {
          pooled[k] += x[row * d + k] / n;
          pooled[d + k] += x[row * d + k] * poolWeights[row];
        }
      }

      const hidden = geluInPlace(linear(pooled, 1, 2 * d, t.proj1_weight, t.proj1_bias, d));
      const vector = linear(hidden, 1, d, t.proj2_weight, t.proj2_bias, this.embedDim);
      let norm = 0;
      for (let k = 0; k < vector.length; k++) {
        norm += vector[k] * vector[k];
      }
      norm = Math.max(Math.sqrt(norm), 1e-12);
      for (let k = 0; k < vector.length; k++) {
        vector[k] /= norm;
      }
      return vector;
    }

    _head(vector, name, labels, squash) {
      const t = this.tensors;
      const logits = linear(
        vector, 1, this.embedDim, t[name + "_weight"], t[name + "_bias"], labels.length
      );
      const scores = squash(logits);
      return labels
        .map((label, index) => ({ label: label, score: scores[index], index: index }))
        .sort((first, second) => second.score - first.score);
    }

    /*
     * What the model reads off an embedding.  Accords are independent 0..1
     * scores (several can be high); family and subfamily each sum to 1.
     * All three lists are sorted, strongest first.
     */
    describe(vector) {
      const sigmoid = (logits) => logits.map((value) => 1 / (1 + Math.exp(-value)));
      return {
        accords: this._head(vector, "accord", this.manifest.accords, sigmoid),
        families: this._head(vector, "family", this.manifest.families, softmaxInPlace),
        subfamilies: this._head(vector, "subfamily", this.manifest.subfamilies, softmaxInPlace),
      };
    }

    /*
     * Recompute the reference pyramids stored in model.json (outputs of the
     * PyTorch model) and report how far this port is from them.
     */
    selfCheck() {
      let minCosine = 1;
      let maxScoreError = 0;
      for (const reference of this.manifest.reference) {
        const result = this.embed(reference.pyramid);
        minCosine = Math.min(minCosine, cosine(result.vector, reference.embedding));
        const described = this.describe(result.vector);
        for (const accord of described.accords) {
          maxScoreError = Math.max(
            maxScoreError, Math.abs(accord.score - reference.accords[accord.index])
          );
        }
        for (const family of described.families) {
          maxScoreError = Math.max(
            maxScoreError, Math.abs(family.score - reference.family[family.index])
          );
        }
      }
      return {
        cases: this.manifest.reference.length,
        minCosine: minCosine,
        maxScoreError: maxScoreError,
        ok: minCosine > 0.9995 && maxScoreError < 0.01,
      };
    }
  }

  async function load(baseUrl) {
    const base = baseUrl || "./";
    const manifestResponse = await fetch(base + "model.json", { cache: "no-cache" });
    if (!manifestResponse.ok) {
      throw new Error("model.json: HTTP " + manifestResponse.status);
    }
    const manifest = await manifestResponse.json();
    // The hash in the URL keeps a cached model.bin from pairing with a newer model.json.
    const binaryUrl = base + manifest.binary + "?v=" + manifest.binary_sha256.slice(0, 12);
    const binaryResponse = await fetch(binaryUrl);
    if (!binaryResponse.ok) {
      throw new Error("model.bin: HTTP " + binaryResponse.status);
    }
    return new Model(manifest, await binaryResponse.arrayBuffer());
  }

  root.ScentVec = { load: load, Model: Model, cosine: cosine };
})(typeof window !== "undefined" ? window : globalThis);
