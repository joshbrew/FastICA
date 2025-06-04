// icaWorker.ts   (this will be the Web Worker file)

/// <reference lib="webworker" />

// ──────────────────────────────────────────────────────────
// 1) Core utilities (typed arrays)
// ──────────────────────────────────────────────────────────

/** Compute the mean of a Float32Array. */
function mean1D(arr: Float32Array): number {
    let sum = 0;
    for (let i = 0; i < arr.length; i++) sum += arr[i];
    return sum / arr.length;
  }
  
  /** Subtract the mean, in‐place, returning a new Float32Array. */
  function subtractMean1D(arr: Float32Array): Float32Array {
    const μ = mean1D(arr);
    const out = new Float32Array(arr.length);
    for (let i = 0; i < arr.length; i++) out[i] = arr[i] - μ;
    return out;
  }
  
  /**
   * Pearson correlation between two Float32Arrays.
   * Both must have the same length.
   */
  function pearsonCorr(x: Float32Array, y: Float32Array): number {
    const N = x.length;
    const μx = mean1D(x);
    const μy = mean1D(y);
    let num = 0, sx = 0, sy = 0;
    for (let i = 0; i < N; i++) {
      const dx = x[i] - μx;
      const dy = y[i] - μy;
      num += dx * dy;
      sx  += dx * dx;
      sy  += dy * dy;
    }
    return num / (Math.sqrt(sx * sy) || 1e-12);
  }
  
  /**
   * Multiply two matrices A (m×k) and B (k×n), stored as 1D Float32Arrays
   * in row‐major form.  A_flat.length = m*k,  B_flat.length = k*n.
   * Returns a new Float32Array of length m*n (row‐major).
   */
  function matrixMultiply(
    A_flat: Float32Array, m: number, k: number,
    B_flat: Float32Array,   n: number
  ): Float32Array {
    // A is m×k, B is k×n ⇒ result is m×n
    const R = new Float32Array(m * n);
    for (let i = 0; i < m; i++) {
      const baseA = i * k;
      const baseR = i * n;
      for (let j = 0; j < n; j++) {
        let acc = 0;
        for (let t = 0; t < k; t++) {
          acc += A_flat[baseA + t] * B_flat[t * n + j];
        }
        R[baseR + j] = acc;
      }
    }
    return R;
  }
  
  /**
   * Transpose a matrix M (rows×cols), stored as Float32Array row‐major.
   * Returns a new Float32Array of size (cols×rows).
   */
  function matrixTranspose(
    M_flat: Float32Array, rows: number, cols: number
  ): Float32Array {
    const T = new Float32Array(rows * cols);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        T[c * rows + r] = M_flat[r * cols + c];
      }
    }
    return T;
  }
  
  /**
   * Compute the covariance matrix of centered data Xc (n×N),
   * where Xc is a Float32Array of length n*N, row‐major (rows = variables).
   * Returns a new Float32Array of length n*n (row‐major).
   */
  function covMatrix(
    Xc_flat: Float32Array, n: number, N: number
  ): Float32Array {
    const C = new Float32Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        let sum = 0;
        const base_i = i * N;
        const base_j = j * N;
        for (let k = 0; k < N; k++) {
          sum += Xc_flat[base_i + k] * Xc_flat[base_j + k];
        }
        C[i * n + j] = sum / N;
      }
    }
    return C;
  }
  
  // ──────────────────────────────────────────────────────────
  // 2) PCA rank-1 reconstruction (n=1) for two-channel mixture
  // ──────────────────────────────────────────────────────────
  
  /**
   * PCA rank-1 for 2 channels.  signals: an array of two Float32Arrays (both length N).
   * Returns two Float32Arrays of length N.
   */
  function pcaReconstruct1(
    signals: [Float32Array, Float32Array]
  ): { rec0: Float32Array; rec1: Float32Array } {
    const [sig0, sig1] = signals;
    const N = sig0.length;
  
    // (a) Center each channel
    const Xc0 = subtractMean1D(sig0);
    const Xc1 = subtractMean1D(sig1);
    const μ0 = mean1D(sig0);
    const μ1 = mean1D(sig1);
  
    // (b) Covariance entries
    let s00 = 0, s11 = 0, s01 = 0;
    for (let k = 0; k < N; k++) {
      const v0 = Xc0[k], v1 = Xc1[k];
      s00 += v0 * v0;
      s11 += v1 * v1;
      s01 += v0 * v1;
    }
    const cov00 = s00 / N;
    const cov11 = s11 / N;
    const cov01 = s01 / N;
  
    // (c) 2×2 principal eigenvector
    const tr = cov00 + cov11;
    const det = cov00 * cov11 - cov01 * cov01;
    const discrim = Math.sqrt(Math.max(tr * tr - 4 * det, 0));
    const λ1 = 0.5 * (tr + discrim);
    let e1x: number, e1y: number;
    if (Math.abs(cov01) > 1e-12) {
      e1x = cov01;
      e1y = λ1 - cov00;
    } else {
      if (cov00 >= cov11) {
        e1x = 1; e1y = 0;
      } else {
        e1x = 0; e1y = 1;
      }
    }
    const nrm = Math.hypot(e1x, e1y) || 1e-12;
    e1x /= nrm;
    e1y /= nrm;
  
    // (d) Project Xc onto e1
    const scores = new Float32Array(N);
    for (let k = 0; k < N; k++) {
      scores[k] = e1x * Xc0[k] + e1y * Xc1[k];
    }
  
    // (e) Reconstruct + add means back
    const rec0 = new Float32Array(N);
    const rec1 = new Float32Array(N);
    for (let k = 0; k < N; k++) {
      rec0[k] = scores[k] * e1x + μ0;
      rec1[k] = scores[k] * e1y + μ1;
    }
    return { rec0, rec1 };
  }
  
  // ──────────────────────────────────────────────────────────
  // 3) Eigen-decomposition (power iteration + deflation)
  // ──────────────────────────────────────────────────────────
  
  /**
   * eigDecompose on symmetric Float32Array (n×n, row‐major).
   * Returns { vals: Float32Array(n), vecs: Float32Array(n×n) (each column = eigenvector) }.
   */
  function eigDecompose(
    C_flat: Float32Array, n: number
  ): { vals: Float32Array; vecs: Float32Array } {
    // We copy C into a local Float32Array A_flat for destructive deflation
    const A_flat = new Float32Array(n * n);
    A_flat.set(C_flat);
  
    const vals = new Float32Array(n);
    const vecs = new Float32Array(n * n);
  
    for (let m = 0; m < n; m++) {
      // initialize random b (length n)
      const b = new Float32Array(n);
      for (let i = 0; i < n; i++) b[i] = Math.random();
  
      // Deflate against previous eigenvectors
      for (let p = 0; p < m; p++) {
        // vecs[ p * n + 0…n-1 ] is the p-th eigenvector
        let dot = 0;
        for (let i = 0; i < n; i++) dot += b[i] * vecs[p * n + i];
        for (let i = 0; i < n; i++) b[i] -= dot * vecs[p * n + i];
      }
  
      let λold = 0;
      // power iteration
      for (let it = 0; it < 1000; it++) {
        // compute A_flat * b → bnew
        const bnew = new Float32Array(n);
        for (let i = 0; i < n; i++) {
          let acc = 0;
          const base = i * n;
          for (let j = 0; j < n; j++) {
            acc += A_flat[base + j] * b[j];
          }
          bnew[i] = acc;
        }
  
        // re‐orthogonalize vs prev eigenvectors
        for (let p = 0; p < m; p++) {
          let dot = 0;
          for (let i = 0; i < n; i++) dot += bnew[i] * vecs[p * n + i];
          for (let i = 0; i < n; i++) bnew[i] -= dot * vecs[p * n + i];
        }
  
        // normalize bnew → b
        let norm = 0;
        for (let i = 0; i < n; i++) norm += bnew[i] * bnew[i];
        norm = Math.sqrt(norm) || 1e-12;
        for (let i = 0; i < n; i++) b[i] = bnew[i] / norm;
  
        // Rayleigh quotient
        let λnew = 0;
        for (let i = 0; i < n; i++) {
          let acc = 0;
          const base = i * n;
          for (let j = 0; j < n; j++) {
            acc += A_flat[base + j] * b[j];
          }
          λnew += b[i] * acc;
        }
        if (Math.abs(λnew - λold) < 1e-8) break;
        λold = λnew;
      }
  
      // exact eigenvalue: b^T (A_flat * b)
      const Ab2 = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        let acc = 0;
        const base = i * n;
        for (let j = 0; j < n; j++) {
          acc += A_flat[base + j] * b[j];
        }
        Ab2[i] = acc;
      }
      let eVal = 0;
      for (let i = 0; i < n; i++) eVal += b[i] * Ab2[i];
  
      // save
      vals[m] = eVal;
      for (let i = 0; i < n; i++) vecs[m * n + i] = b[i];
  
      // deflate: A_flat ← A_flat − eVal * (b b^T)
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          A_flat[i * n + j] -= eVal * b[i] * b[j];
        }
      }
    }
  
    return { vals, vecs };
  }
  
  // ──────────────────────────────────────────────────────────
  // 4) Whitening for n×N
  // ──────────────────────────────────────────────────────────
  
  /**
   * Whiten the centered data Xc (n×N) where Xc is Float32Array row‐major.
   * Returns { Xw: Float32Array(n×N), E: Float32Array(n×n), D: Float32Array(n) }.
   */
  function whiten(
    Xc_flat: Float32Array, n: number, N: number
  ): { Xw: Float32Array; E: Float32Array; D: Float32Array } {
    // compute covariance (n×n)
    const C_flat = covMatrix(Xc_flat, n, N);
  
    // eigendecompose
    const { vals: D, vecs } = eigDecompose(C_flat, n);
  
    // build E (n×n): columns are vecs[m*n … m*n + n−1]
    const E_flat = new Float32Array(n * n);
    for (let col = 0; col < n; col++) {
      for (let row = 0; row < n; row++) {
        E_flat[row * n + col] = vecs[col * n + row];
      }
    }
  
    // DinvSqrt
    const DinvSqrt = new Float32Array(n);
    for (let i = 0; i < n; i++) DinvSqrt[i] = 1 / Math.sqrt(D[i] || 1e-12);
  
    // ET = transpose(E_flat)
    const ET_flat = new Float32Array(n * n);
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        ET_flat[c * n + r] = E_flat[r * n + c];
      }
    }
  
    // ETXc (n×N) = ET (n×n) · Xc (n×N)
    const ETXc_flat = matrixMultiply(ET_flat, n, n, Xc_flat, N);
  
    // scale each row i of ETXc by DinvSqrt[i] to form Xw
    const Xw_flat = new Float32Array(n * N);
    for (let i = 0; i < n; i++) {
      const scale = DinvSqrt[i];
      const base_i = i * N;
      for (let k = 0; k < N; k++) {
        Xw_flat[base_i + k] = ETXc_flat[base_i + k] * scale;
      }
    }
  
    return { Xw: Xw_flat, E: E_flat, D };
  }
  
  // ──────────────────────────────────────────────────────────
  // 5) General symmetric FastICA (n×n)
  // ──────────────────────────────────────────────────────────
  
  /**
   * fastICA_n on n×N Float32Array data.  Returns ICs (n×N), mixingMat (n×n), demixMat (n×n), reconstructed (n×N).
   */
  function fastICA_n(
    X_flat: Float32Array, n: number, N: number,
    tol: number = 1e-6, maxIter: number = 200
  ): {
    ICs: Float32Array;
    mixingMat: Float32Array;
    demixMat: Float32Array;
    reconstructed: Float32Array;
  } {
    // (a) Center each row (n×N)
    const Xc_flat = new Float32Array(n * N);
    for (let i = 0; i < n; i++) {
      const base_i = i * N;
      // compute row mean
      let rowSum = 0;
      for (let k = 0; k < N; k++) rowSum += X_flat[base_i + k];
      const μ = rowSum / N;
      for (let k = 0; k < N; k++) {
        Xc_flat[base_i + k] = X_flat[base_i + k] - μ;
      }
    }
  
    // (b) Whiten
    const { Xw: Xw_flat, E: E_flat, D } = whiten(Xc_flat, n, N);
  
    // (c) Initialize random W (n×n) with orthonormal rows
    let W_flat = new Float32Array(n * n);
    for (let i = 0; i < n; i++) {
      // random row
      let norm = 0;
      for (let j = 0; j < n; j++) {
        const v = Math.random();
        W_flat[i * n + j] = v;
        norm += v * v;
      }
      norm = Math.sqrt(norm) || 1e-12;
      for (let j = 0; j < n; j++) {
        W_flat[i * n + j] /= norm;
      }
    }
  
    /** Orthogonalize rows of W (n×n). */
    function orthogonalize_rows(M_flat: Float32Array): Float32Array {
      const MMT_flat = new Float32Array(n * n);
      // compute M·M^T  (n×n)
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          let sum = 0;
          for (let k = 0; k < n; k++) {
            sum += M_flat[i * n + k] * M_flat[j * n + k];
          }
          MMT_flat[i * n + j] = sum;
        }
      }
      // eigendecompose MMT_flat
      const { vals: vMMT, vecs: uMMT_flat } = eigDecompose(MMT_flat, n);
      // build U from uMMT_flat
      const U_flat = new Float32Array(n * n);
      for (let col = 0; col < n; col++) {
        for (let row = 0; row < n; row++) {
          U_flat[row * n + col] = uMMT_flat[col * n + row];
        }
      }
      // invSqrt = 1/√(vMMT[i])
      const invSqrt = new Float32Array(n);
      for (let i = 0; i < n; i++) invSqrt[i] = 1 / Math.sqrt(vMMT[i] || 1e-12);
  
      // UD = U · diag(invSqrt)
      const UD_flat = new Float32Array(n * n);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          UD_flat[i * n + j] = U_flat[i * n + j] * invSqrt[j];
        }
      }
  
      // Mcorr = UD · U^T  (n×n)
      const Mcorr_flat = matrixMultiply(UD_flat, n, n, matrixTranspose(U_flat, n, n), n);
  
      // return Mcorr · M
      return matrixMultiply(Mcorr_flat, n, n, M_flat, n);
    }
  
    // orthonormalize initial W
    W_flat = orthogonalize_rows(W_flat);
  
    // (d) FastICA iterations (symmetric)
    const WXw_flat = new Float32Array(n * N);
    const Y_flat    = new Float32Array(n * N);
    const G_flat    = new Float32Array(n * N);
    const Gp_flat   = new Float32Array(n * N);
    const Wnew_flat = new Float32Array(n * n);
  
    for (let iter = 0; iter < maxIter; iter++) {
      // save old W
      const Wold_flat = W_flat.slice();
  
      // 1) Y = W · Xw  (n×N)
      for (let i = 0; i < n; i++) {
        const baseW_i = i * n;
        const baseY_i = i * N;
        for (let k = 0; k < N; k++) {
          let acc = 0;
          for (let j = 0; j < n; j++) {
            acc += W_flat[baseW_i + j] * Xw_flat[j * N + k];
          }
          Y_flat[baseY_i + k] = acc;
        }
      }
  
      // 2) G = tanh(Y), Gp = 1−tanh^2(Y)
      for (let i = 0; i < n; i++) {
        const base = i * N;
        for (let k = 0; k < N; k++) {
          const t = Math.tanh(Y_flat[base + k]);
          G_flat[base + k]  = t;
          Gp_flat[base + k] = 1 - t * t;
        }
      }
  
      // 3) Compute Wnew: for each row i,
      //    α = mean of Gp[i,*], β_j = mean over k of Xw[j,k] * G[i,k]
      for (let i = 0; i < n; i++) {
        // α
        let α = 0;
        {
          const baseGp = i * N;
          for (let k = 0; k < N; k++) α += Gp_flat[baseGp + k];
        }
        α /= N;
  
        // β_j for j=0..n−1
        const β = new Float32Array(n);
        for (let j = 0; j < n; j++) {
          let sum = 0;
          const baseXwj = j * N;
          const baseG_i = i * N;
          for (let k = 0; k < N; k++) {
            sum += Xw_flat[baseXwj + k] * G_flat[baseG_i + k];
          }
          β[j] = sum / N;
        }
  
        // wnew_i = β − α * W_i
        const baseW_i = i * n;
        const baseWnew_i = i * n;
        for (let j = 0; j < n; j++) {
          Wnew_flat[baseWnew_i + j] = β[j] - α * W_flat[baseW_i + j];
        }
      }
  
      // 4) orthonormalize Wnew_flat ⇒ W_flat
      W_flat = orthogonalize_rows(Wnew_flat);
  
      // 5) check convergence: max | |wnew·wold| − 1 |
      let maxDiff = 0;
      for (let i = 0; i < n; i++) {
        let dot = 0;
        const base_old = i * n;
        const base_new = i * n;
        for (let j = 0; j < n; j++) {
          dot += Wold_flat[base_old + j] * W_flat[base_new + j];
        }
        maxDiff = Math.max(maxDiff, Math.abs(Math.abs(dot) - 1));
      }
      if (maxDiff < tol) break;
    }
  
    // (e) Independent components: S = W · Xw (n×N)
    const S_flat = new Float32Array(n * N);
    for (let i = 0; i < n; i++) {
      const baseW_i = i * n;
      const baseS_i = i * N;
      for (let k = 0; k < N; k++) {
        let acc = 0;
        for (let j = 0; j < n; j++) {
          acc += W_flat[baseW_i + j] * Xw_flat[j * N + k];
        }
        S_flat[baseS_i + k] = acc;
      }
    }
  
    // (f) Recover mixing matrix A = E · D^{1/2} · W^{-1}
    function invertFlatNxN(M_flat: Float32Array, n: number): Float32Array {
      // same Gaussian elimination on Float32Array (n×n)
      const A_flat = new Float32Array(n * n);
      A_flat.set(M_flat);
      const I_flat = new Float32Array(n * n);
      for (let i = 0; i < n; i++) I_flat[i * n + i] = 1;
      const aug = new Float32Array(n * (2 * n));
      // build augmented [A | I]
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          aug[i * (2 * n) + j] = A_flat[i * n + j];
          aug[i * (2 * n) + (n + j)] = I_flat[i * n + j];
        }
      }
  
      // forward elimination + back substitution
      for (let i = 0; i < n; i++) {
        // pivot
        let maxRow = i;
        for (let r = i + 1; r < n; r++) {
          if (Math.abs(aug[r * (2 * n) + i]) > Math.abs(aug[maxRow * (2 * n) + i])) {
            maxRow = r;
          }
        }
        if (Math.abs(aug[maxRow * (2 * n) + i]) < 1e-12) {
          throw new Error("Singular matrix");
        }
        // swap rows i & maxRow
        if (maxRow !== i) {
          for (let c = 0; c < 2 * n; c++) {
            const tmp = aug[i * (2 * n) + c];
            aug[i * (2 * n) + c] = aug[maxRow * (2 * n) + c];
            aug[maxRow * (2 * n) + c] = tmp;
          }
        }
        // normalize row i
        const pivot = aug[i * (2 * n) + i];
        for (let c = 0; c < 2 * n; c++) {
          aug[i * (2 * n) + c] /= pivot;
        }
        // eliminate other rows
        for (let r = 0; r < n; r++) {
          if (r === i) continue;
          const factor = aug[r * (2 * n) + i];
          if (factor === 0) continue;
          for (let c = 0; c < 2 * n; c++) {
            aug[r * (2 * n) + c] -= factor * aug[i * (2 * n) + c];
          }
        }
      }
  
      // the right half (columns n…2n−1) is now the inverse
      const M_inv_flat = new Float32Array(n * n);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          M_inv_flat[i * n + j] = aug[i * (2 * n) + (n + j)];
        }
      }
      return M_inv_flat;
    }
  
    // build D^{1/2}
    const Dsqrt_flat = new Float32Array(n * n);
    for (let i = 0; i < n; i++) {
      Dsqrt_flat[i * n + i] = Math.sqrt(D[i] || 0);
    }
  
    // invert W_flat
    const Winv_flat = invertFlatNxN(W_flat, n);
  
    // compute A_flat = E_flat · Dsqrt_flat · Winv_flat
    const EDsqrt_flat  = matrixMultiply(E_flat, n, n, Dsqrt_flat, n);
    const A_flat      = matrixMultiply(EDsqrt_flat, n, n, Winv_flat, n);
  
    // (g) Reconstruct Xc_rec = A·S  (n×N), then add means back
    const Xc_rec_flat = matrixMultiply(A_flat, n, n, S_flat, N);
    // compute μ_i for each row of original X
    const μ = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let sum = 0;
      const base = i * N;
      for (let k = 0; k < N; k++) sum += X_flat[base + k];
      μ[i] = sum / N;
    }
    // add μ[i] back to each row
    const Xrec_flat = new Float32Array(n * N);
    for (let i = 0; i < n; i++) {
      const baseXc = i * N;
      for (let k = 0; k < N; k++) {
        Xrec_flat[baseXc + k] = Xc_rec_flat[baseXc + k] + μ[i];
      }
    }
  
    return {
      ICs: S_flat,
      mixingMat: A_flat,
      demixMat: W_flat,
      reconstructed: Xrec_flat
    };
  }
  
  // ──────────────────────────────────────────────────────────
  // 6) fastICA_2x2 as special case
  // ──────────────────────────────────────────────────────────
  
  /**
   * fastICA specialized for 2×2 mixtures, on Float32Arrays of length N.
   * Returns ICs (2×N), mixingMat (2×2), demixMat (2×2), recSignal1 & recSignal2.
   */
  function fastICA_2x2(
    signals: [Float32Array, Float32Array],
    tol = 1e-6, maxIter = 200
  ): {
    ICs: Float32Array;
    mixingMat: Float32Array;
    demixMat: Float32Array;
    recSignal1: Float32Array;
    recSignal2: Float32Array;
  } {
    const [sig0, sig1] = signals;
    const N = sig0.length;
  
    // (a) Center
    const Xc0 = subtractMean1D(sig0);
    const Xc1 = subtractMean1D(sig1);
    const μ0 = mean1D(sig0);
    const μ1 = mean1D(sig1);
  
    // (b) Covariance entries
    let s00 = 0, s11 = 0, s01 = 0;
    for (let k = 0; k < N; k++) {
      s00 += Xc0[k] * Xc0[k];
      s11 += Xc1[k] * Xc1[k];
      s01 += Xc0[k] * Xc1[k];
    }
    const cov00 = s00 / N;
    const cov11 = s11 / N;
    const cov01 = s01 / N;
  
    // (c) 2×2 eigendecomposition
    const tr = cov00 + cov11;
    const det = cov00 * cov11 - cov01 * cov01;
    const discrim = Math.sqrt(Math.max(tr * tr - 4 * det, 0));
    const λ1 = 0.5 * (tr + discrim);
    const λ2 = 0.5 * (tr - discrim);
  
    // eigenvector for λ1
    let ex: number, ey: number;
    if (Math.abs(cov01) > 1e-12) {
      ex = cov01;
      ey = λ1 - cov00;
    } else {
      if (cov00 >= cov11) {
        ex = 1; ey = 0;
      } else {
        ex = 0; ey = 1;
      }
    }
    const nrm = Math.hypot(ex, ey) || 1e-12;
    ex /= nrm;
    ey /= nrm;
  
    // second eigenvector orthonormal
    const fx = -ey;
    const fy = ex;
  
    // E = [ [ex, fx], [ey, fy] ]    (2×2)
    const E_flat = new Float32Array([ex, fx, ey, fy]);
  
    // D^{-1/2}
    const DinvSqrt0 = 1 / Math.sqrt(λ1 || 1e-12);
    const DinvSqrt1 = 1 / Math.sqrt(λ2 || 1e-12);
  
    // (d) Whiten: Xw = D^{-1/2} · E^T · Xc
    // E^T = [ [ex, ey], [fx, fy] ]  (2×2)
    // Xc (2×N) row‐major → [Xc0, Xc1]
    const Xw_flat = new Float32Array(2 * N);
    for (let k = 0; k < N; k++) {
      const x0 = Xc0[k], x1 = Xc1[k];
      const dot1 = ex * x0 + ey * x1;
      const dot2 = fx * x0 + fy * x1;
      Xw_flat[0 * N + k] = dot1 * DinvSqrt0;
      Xw_flat[1 * N + k] = dot2 * DinvSqrt1;
    }
  
    // (e) initialize random demixing columns wCols
    const wCols0 = new Float32Array([Math.random(), Math.random()]);
    const wCols1 = new Float32Array([Math.random(), Math.random()]);
    // normalize them
    {
      const n0 = Math.hypot(wCols0[0], wCols0[1]) || 1e-12;
      wCols0[0] /= n0; wCols0[1] /= n0;
    }
    {
      const n1 = Math.hypot(wCols1[0], wCols1[1]) || 1e-12;
      wCols1[0] /= n1; wCols1[1] /= n1;
    }
  
    // We'll store demixing matrix as row‐major 2×2: W_flat = [ wCols0[0], wCols0[1],  wCols1[0], wCols1[1] ]
    let W_flat = new Float32Array([
      wCols0[0], wCols0[1],
      wCols1[0], wCols1[1]
    ]);
  
    // do deflationary FastICA
    for (let ic = 0; ic < 2; ic++) {
      // pick column ic: either W_flat[0..1] or W_flat[2..3]
      let wX = new Float32Array(2);
      wX[0] = W_flat[ic * 2 + 0];
      wX[1] = W_flat[ic * 2 + 1];
  
      for (let iter = 0; iter < maxIter; iter++) {
        // 1) w^T · Xw row‐by‐row
        const wtx = new Float32Array(N);
        for (let k = 0; k < N; k++) {
          wtx[k] = wX[0] * Xw_flat[0 * N + k] + wX[1] * Xw_flat[1 * N + k];
        }
  
        // 2) g(u) = tanh(u), g'(u) = 1 − tanh^2(u)
        const gw = new Float32Array(N), gPw = new Float32Array(N);
        for (let k = 0; k < N; k++) {
          const t = Math.tanh(wtx[k]);
          gw[k] = t;
          gPw[k] = 1 - t * t;
        }
  
        // 3) update
        let sumGp = 0;
        for (let k = 0; k < N; k++) sumGp += gPw[k];
        sumGp /= N;
  
        const wNew = new Float32Array(2);
        for (let i = 0; i < 2; i++) {
          let acc = 0;
          const baseXw_i = i * N;
          for (let k = 0; k < N; k++) {
            acc += Xw_flat[baseXw_i + k] * gw[k];
          }
          wNew[i] = acc / N - sumGp * wX[i];
        }
  
        // 4) deflation (if ic>0 subtract projection onto previous w)
        if (ic > 0) {
          // previous column is W_flat[0..1]
          const w0 = new Float32Array([ W_flat[0], W_flat[1] ]);
          const proj = wNew[0] * w0[0] + wNew[1] * w0[1];
          wNew[0] -= proj * w0[0];
          wNew[1] -= proj * w0[1];
        }
  
        // 5) normalize
        const nNew = Math.hypot(wNew[0], wNew[1]) || 1e-12;
        wNew[0] /= nNew;
        wNew[1] /= nNew;
  
        // 6) convergence: | |wNew·wX| − 1 | < tol
        const corr = wNew[0] * wX[0] + wNew[1] * wX[1];
        if (Math.abs(Math.abs(corr) - 1) < tol) {
          wX[0] = wNew[0];
          wX[1] = wNew[1];
          break;
        }
        wX[0] = wNew[0];
        wX[1] = wNew[1];
      }
  
      // write back into W_flat
      W_flat[ic * 2 + 0] = wX[0];
      W_flat[ic * 2 + 1] = wX[1];
    }
  
    // (g) Independent components: S_flat = W_flat · Xw_flat  (2×N)
    const S_flat = new Float32Array(2 * N);
    for (let i = 0; i < 2; i++) {
      const baseW_i = i * 2;
      const baseS_i = i * N;
      for (let k = 0; k < N; k++) {
        S_flat[baseS_i + k] = W_flat[baseW_i + 0] * Xw_flat[0 * N + k]
                            + W_flat[baseW_i + 1] * Xw_flat[1 * N + k];
      }
    }
  
    // (h) Reconstruct mixing matrix A = E · D^{1/2} · W^{-1}
  
    /** Invert 2×2 Float32Array (row‐major). */
    function invert2x2(M2_flat: Float32Array): Float32Array {
      const a = M2_flat[0], b = M2_flat[1], c = M2_flat[2], d = M2_flat[3];
      const det = a * d - b * c || 1e-12;
      return new Float32Array([d / det, -b / det, -c / det, a / det]);
    }
  
    // D^{1/2} = diag(sqrt(λ1), sqrt(λ2))
    const Dsqrt_flat = new Float32Array([Math.sqrt(λ1 || 0), 0, 0, Math.sqrt(λ2 || 0)]);
  
    // Winv
    const Winv_flat = invert2x2(W_flat);
  
    // EDsqrt = E_flat (2×2) · Dsqrt_flat (2×2)
    const EDsqrt_flat = new Float32Array(4);
    // 2×2 multiply:
    EDsqrt_flat[0] = E_flat[0] * Dsqrt_flat[0] + E_flat[1] * Dsqrt_flat[2];
    EDsqrt_flat[1] = E_flat[0] * Dsqrt_flat[1] + E_flat[1] * Dsqrt_flat[3];
    EDsqrt_flat[2] = E_flat[2] * Dsqrt_flat[0] + E_flat[3] * Dsqrt_flat[2];
    EDsqrt_flat[3] = E_flat[2] * Dsqrt_flat[1] + E_flat[3] * Dsqrt_flat[3];
  
    // A_flat = EDsqrt_flat · Winv_flat  (2×2 × 2×2)
    const A_flat = new Float32Array(4);
    A_flat[0] = EDsqrt_flat[0]*Winv_flat[0] + EDsqrt_flat[1]*Winv_flat[2];
    A_flat[1] = EDsqrt_flat[0]*Winv_flat[1] + EDsqrt_flat[1]*Winv_flat[3];
    A_flat[2] = EDsqrt_flat[2]*Winv_flat[0] + EDsqrt_flat[3]*Winv_flat[2];
    A_flat[3] = EDsqrt_flat[2]*Winv_flat[1] + EDsqrt_flat[3]*Winv_flat[3];
  
    // (i) Reconstruct: Xc_rec_flat = A_flat (2×2) · S_flat (2×N)
    const Xc_rec_flat = new Float32Array(2 * N);
    for (let k = 0; k < N; k++) {
      // channel 0: row 0 of A_flat = [A_flat[0], A_flat[1]]
      Xc_rec_flat[0 * N + k] = A_flat[0] * S_flat[0 * N + k] + A_flat[1] * S_flat[1 * N + k];
      // channel 1: row 1 of A_flat = [A_flat[2], A_flat[3]]
      Xc_rec_flat[1 * N + k] = A_flat[2] * S_flat[0 * N + k] + A_flat[3] * S_flat[1 * N + k];
    }
  
    // add means back
    const rec0 = new Float32Array(N);
    const rec1 = new Float32Array(N);
    for (let k = 0; k < N; k++) {
      rec0[k] = Xc_rec_flat[0 * N + k] + μ0;
      rec1[k] = Xc_rec_flat[1 * N + k] + μ1;
    }
  
    return {
      ICs:      S_flat,
      mixingMat: A_flat,
      demixMat: W_flat,
      recSignal1: rec0,
      recSignal2: rec1
    };
  }
  
  // ──────────────────────────────────────────────────────────
  // 7) Estimate dominant frequency (small DFT)
  // ──────────────────────────────────────────────────────────
  
  /**
   * Estimate dominant frequency (naive DFT).
   */
  function estimateDominantFreq(
    signal: Float32Array, fs: number
  ): number {
    const N = signal.length;
    const half = Math.floor(N / 2);
    let bestF = 0, bestMag = 0;
    for (let k = 1; k <= half; k++) {
      let re = 0, im = 0;
      const ω = (2 * Math.PI * k) / N;
      for (let n = 0; n < N; n++) {
        const v = signal[n];
        re += v * Math.cos(ω * n);
        im -= v * Math.sin(ω * n);
      }
      const mag = Math.hypot(re, im);
      if (mag > bestMag) {
        bestMag = mag;
        bestF = (k * fs) / N;
      }
    }
    return bestF;
  }
  
  // ──────────────────────────────────────────────────────────
  // 8) Worker message handling
  // ──────────────────────────────────────────────────────────
  
  interface MessageIn {
    type: "run2x2" | "runNxM";
    data: Float32Array[];     // array of Float32Array channels (length = n)
    n: number;                // number of rows (channels)
    N: number;                // number of samples per channel
    fs?: number;              // sampling rate
  }
  
  interface MessageOut {
    type: "result2x2" | "resultNxN";
    ICs: Float32Array;
    mixingMat: Float32Array;
    demixMat: Float32Array;
    reconstructed: Float32Array;
    corrs?: Float32Array;     // if NxN we can send back per‐channel correlations
    domFreqs?: Float32Array;  // dominant freqs of ICs
  }
  
  self.addEventListener("message", (evt: MessageEvent<MessageIn>) => {
    const msg = evt.data;
  
    if (msg.type === "run2x2") {
      const [ch0, ch1] = msg.data;
      const result = fastICA_2x2([ch0, ch1]);
      const fsamp = msg.fs || 1;
      // estimate dom freqs
      const freqs = new Float32Array(2);
      freqs[0] = estimateDominantFreq(result.ICs.subarray(0 * msg.N, 1 * msg.N), fsamp);
      freqs[1] = estimateDominantFreq(result.ICs.subarray(1 * msg.N, 2 * msg.N), fsamp);
  
      const out: MessageOut = {
        type: "result2x2",
        ICs: result.ICs,
        mixingMat: result.mixingMat,
        demixMat: result.demixMat,
        reconstructed: new Float32Array(msg.N).map((_, i) => result.recSignal1[i]), 
        domFreqs: freqs
        // note: recSignal2 also available if needed
      };
      // Transfer all large buffers back
      const transfer = [
        out.ICs.buffer,
        out.mixingMat.buffer,
        out.demixMat.buffer,
        out.reconstructed.buffer
      ];

      if(out.domFreqs) transfer.push(
        out.domFreqs.buffer
        );

      (self as any).postMessage(out, transfer);
    }
  
    if (msg.type === "runNxM") {
      // data: array of n Float32Arrays each length N
      const n = msg.n, N = msg.N;
      // stack them row‐major into one Float32Array
      const X_flat = new Float32Array(n * N);
      for (let i = 0; i < n; i++) {
        X_flat.set(msg.data[i], i * N);
      }
      const { ICs, mixingMat, demixMat, reconstructed } = fastICA_n(X_flat, n, N);
      // compute per‐channel correlations
      const corrs = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const origRow = msg.data[i];
        const recRow  = reconstructed.subarray(i * N, (i + 1) * N);
        corrs[i] = pearsonCorr(origRow, recRow);
      }
      // estimate dom freqs of each IC
      const freqs = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        freqs[i] = estimateDominantFreq(ICs.subarray(i * N, (i + 1) * N), msg.fs || 1);
      }
  
      const out: MessageOut = {
        type: "resultNxN",
        ICs,
        mixingMat,
        demixMat,
        reconstructed,
        corrs,
        domFreqs: freqs
      };
      (self as any).postMessage(out, [
        ICs.buffer,
        mixingMat.buffer,
        demixMat.buffer,
        reconstructed.buffer,
        corrs.buffer,
        freqs.buffer
      ]);
    }
  });
  