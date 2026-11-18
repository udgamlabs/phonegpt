/*
phone_gpt.c: train a small GPT from random initialisation on a phone CPU.
Adapted from Andrej Karpathy's llm.c train_gpt2.c (MIT licence, see LICENSE).
Changes: random init (no checkpoint), character-level data, a pthread pool with dynamic
chunks (big.LITTLE friendly) instead of OpenMP, NEON matmul kernel, O(T^2) attention
backward, AdamW with weight-decay mask and grad clipping (nanoGPT style), cosine LR,
fixed held-out eval batches, thermal pause, checkpoints and samples, optional dropout on
the embedding and the two residual branches (-dropout p; nanoGPT also drops attention weights, this does not).
Build: zig cc -target aarch64-linux-musl -O3 -ffast-math -static -o phone_gpt phone_gpt.c -lpthread -lm
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <stdatomic.h>
#include <unistd.h>
#ifdef __aarch64__
#include <arm_neon.h>
#endif

static void* xmalloc(size_t n) { void* p = malloc(n); if (!p) { fprintf(stderr, "malloc %zu failed\n", n); exit(1); } return p; }
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec / 1e9; }

// ----------------------------------------------------------------------------
// thread pool: parallel_for(n, chunk, fn, ctx) runs fn(ctx, s, e) over [0,n) in chunks

typedef void (*range_fn)(void* ctx, int s, int e);
static int g_nthreads = 1;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_cv = PTHREAD_COND_INITIALIZER;
static pthread_cond_t g_done = PTHREAD_COND_INITIALIZER;
static int g_gen = 0, g_active = 0, g_n = 0, g_chunk = 1;
static range_fn g_fn; static void* g_ctx;
static atomic_int g_next;

static void run_chunks(void) {
    for (;;) {
        int s = atomic_fetch_add(&g_next, g_chunk);
        if (s >= g_n) break;
        int e = s + g_chunk; if (e > g_n) e = g_n;
        g_fn(g_ctx, s, e);
    }
}
static void* worker(void* arg) {
    (void)arg; int my_gen = 0;
    for (;;) {
        pthread_mutex_lock(&g_mu);
        while (g_gen == my_gen) pthread_cond_wait(&g_cv, &g_mu);
        my_gen = g_gen;
        pthread_mutex_unlock(&g_mu);
        run_chunks();
        pthread_mutex_lock(&g_mu);
        if (--g_active == 0) pthread_cond_signal(&g_done);
        pthread_mutex_unlock(&g_mu);
    }
    return NULL;
}
static void pool_init(int n) {
    g_nthreads = n;
    for (int i = 0; i < n - 1; i++) { pthread_t t; pthread_create(&t, NULL, worker, NULL); pthread_detach(t); }
}
static void parallel_for(int n, int chunk, range_fn fn, void* ctx) {
    if (chunk < 1) chunk = 1;
    if (g_nthreads <= 1 || n <= chunk) { fn(ctx, 0, n); return; }
    pthread_mutex_lock(&g_mu);
    g_fn = fn; g_ctx = ctx; g_n = n; g_chunk = chunk; atomic_store(&g_next, 0);
    g_active = g_nthreads - 1; g_gen++;
    pthread_cond_broadcast(&g_cv);
    pthread_mutex_unlock(&g_mu);
    run_chunks();
    pthread_mutex_lock(&g_mu);
    while (g_active > 0) pthread_cond_wait(&g_done, &g_mu);
    pthread_mutex_unlock(&g_mu);
}
static int auto_chunk(int n) { int c = n / (g_nthreads * 8); return c < 1 ? 1 : c; }

// ----------------------------------------------------------------------------
// layers. B = batch, T = time, C = channels, V = vocab (Vp padded to a multiple of 4)

static void encoder_forward(float* out, const int* inp, const float* wte, const float* wpe, int B, int T, int C) {
    for (int b = 0; b < B; b++) for (int t = 0; t < T; t++) {
        float* o = out + (size_t)(b * T + t) * C; const float* we = wte + (size_t)inp[b * T + t] * C; const float* wp = wpe + (size_t)t * C;
        for (int i = 0; i < C; i++) o[i] = we[i] + wp[i];
    }
}
static void encoder_backward(float* dwte, float* dwpe, const float* dout, const int* inp, int B, int T, int C) {
    for (int b = 0; b < B; b++) for (int t = 0; t < T; t++) {
        const float* d = dout + (size_t)(b * T + t) * C; float* dwe = dwte + (size_t)inp[b * T + t] * C; float* dwp = dwpe + (size_t)t * C;
        for (int i = 0; i < C; i++) { dwe[i] += d[i]; dwp[i] += d[i]; }
    }
}

typedef struct { float *out, *mean, *rstd; const float *inp, *w, *bias, *dout; float *dinp, *dw, *db; int C, N; } LNCtx;
static void ln_fwd_range(void* p, int s, int e) {
    LNCtx* c = p; int C = c->C;
    for (int r = s; r < e; r++) {
        const float* x = c->inp + (size_t)r * C;
        float m = 0; for (int i = 0; i < C; i++) m += x[i]; m /= C;
        float v = 0; for (int i = 0; i < C; i++) { float d = x[i] - m; v += d * d; } v /= C;
        float rs = 1.0f / sqrtf(v + 1e-5f);
        float* o = c->out + (size_t)r * C;
        for (int i = 0; i < C; i++) o[i] = (x[i] - m) * rs * c->w[i] + c->bias[i];
        c->mean[r] = m; c->rstd[r] = rs;
    }
}
static void layernorm_forward(float* out, float* mean, float* rstd, const float* inp, const float* w, const float* b, int N, int C) {
    LNCtx c = { .out = out, .mean = mean, .rstd = rstd, .inp = inp, .w = w, .bias = b, .C = C, .N = N };
    parallel_for(N, auto_chunk(N), ln_fwd_range, &c);
}
static void ln_bwd_rows(void* p, int s, int e) {
    LNCtx* c = p; int C = c->C;
    for (int r = s; r < e; r++) {
        const float* d = c->dout + (size_t)r * C; const float* x = c->inp + (size_t)r * C; float* dx = c->dinp + (size_t)r * C;
        float m = c->mean[r], rs = c->rstd[r], dm = 0, dnm = 0;
        for (int i = 0; i < C; i++) { float n = (x[i] - m) * rs; float dn = c->w[i] * d[i]; dm += dn; dnm += dn * n; }
        dm /= C; dnm /= C;
        for (int i = 0; i < C; i++) { float n = (x[i] - m) * rs; float dn = c->w[i] * d[i]; dx[i] += (dn - dm - n * dnm) * rs; }
    }
}
static void ln_bwd_cols(void* p, int s, int e) {
    LNCtx* c = p; int C = c->C;
    for (int i = s; i < e; i++) {
        float gw = 0, gb = 0;
        for (int r = 0; r < c->N; r++) {
            float d = c->dout[(size_t)r * C + i]; float n = (c->inp[(size_t)r * C + i] - c->mean[r]) * c->rstd[r];
            gw += n * d; gb += d;
        }
        c->dw[i] += gw; c->db[i] += gb;
    }
}
static void layernorm_backward(float* dinp, float* dw, float* db, const float* dout, const float* inp, const float* w,
                               const float* mean, const float* rstd, int N, int C) {
    LNCtx c = { .dinp = dinp, .dw = dw, .db = db, .dout = dout, .inp = inp, .w = w, .mean = mean, .rstd = rstd, .C = C, .N = N };
    parallel_for(N, auto_chunk(N), ln_bwd_rows, &c);
    parallel_for(C, auto_chunk(C), ln_bwd_cols, &c);
}

// matmul: out (N,OC) = inp (N,C) @ W(OC,C)^T + bias. N % 4 == 0, C % 4 == 0, OC % 4 == 0.
typedef struct { float* out; const float *inp, *w, *bias, *dout; float *dinp, *dw, *db; int N, C, OC; } MMCtx;
static void mm_fwd_range(void* p, int s, int e) {
    MMCtx* c = p; int C = c->C, OC = c->OC;
    for (int blk = s; blk < e; blk++) {
        int r0 = blk * 4;
        const float* x0 = c->inp + (size_t)(r0 + 0) * C; const float* x1 = x0 + C; const float* x2 = x1 + C; const float* x3 = x2 + C;
        for (int o = 0; o < OC; o += 4) {
            const float* w0 = c->w + (size_t)o * C; const float* w1 = w0 + C; const float* w2 = w1 + C; const float* w3 = w2 + C;
            float res[4][4];
#ifdef __aarch64__
            float32x4_t a00 = vdupq_n_f32(0), a01 = a00, a02 = a00, a03 = a00, a10 = a00, a11 = a00, a12 = a00, a13 = a00,
                        a20 = a00, a21 = a00, a22 = a00, a23 = a00, a30 = a00, a31 = a00, a32 = a00, a33 = a00;
            for (int i = 0; i < C; i += 4) {
                float32x4_t X0 = vld1q_f32(x0 + i), X1 = vld1q_f32(x1 + i), X2 = vld1q_f32(x2 + i), X3 = vld1q_f32(x3 + i);
                float32x4_t W0 = vld1q_f32(w0 + i), W1 = vld1q_f32(w1 + i), W2 = vld1q_f32(w2 + i), W3 = vld1q_f32(w3 + i);
                a00 = vfmaq_f32(a00, X0, W0); a01 = vfmaq_f32(a01, X0, W1); a02 = vfmaq_f32(a02, X0, W2); a03 = vfmaq_f32(a03, X0, W3);
                a10 = vfmaq_f32(a10, X1, W0); a11 = vfmaq_f32(a11, X1, W1); a12 = vfmaq_f32(a12, X1, W2); a13 = vfmaq_f32(a13, X1, W3);
                a20 = vfmaq_f32(a20, X2, W0); a21 = vfmaq_f32(a21, X2, W1); a22 = vfmaq_f32(a22, X2, W2); a23 = vfmaq_f32(a23, X2, W3);
                a30 = vfmaq_f32(a30, X3, W0); a31 = vfmaq_f32(a31, X3, W1); a32 = vfmaq_f32(a32, X3, W2); a33 = vfmaq_f32(a33, X3, W3);
            }
            res[0][0] = vaddvq_f32(a00); res[0][1] = vaddvq_f32(a01); res[0][2] = vaddvq_f32(a02); res[0][3] = vaddvq_f32(a03);
            res[1][0] = vaddvq_f32(a10); res[1][1] = vaddvq_f32(a11); res[1][2] = vaddvq_f32(a12); res[1][3] = vaddvq_f32(a13);
            res[2][0] = vaddvq_f32(a20); res[2][1] = vaddvq_f32(a21); res[2][2] = vaddvq_f32(a22); res[2][3] = vaddvq_f32(a23);
            res[3][0] = vaddvq_f32(a30); res[3][1] = vaddvq_f32(a31); res[3][2] = vaddvq_f32(a32); res[3][3] = vaddvq_f32(a33);
#else
            const float* xs[4] = { x0, x1, x2, x3 }; const float* ws[4] = { w0, w1, w2, w3 };
            for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++) { float v = 0; for (int i = 0; i < C; i++) v += xs[a][i] * ws[b][i]; res[a][b] = v; }
#endif
            for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++)
                c->out[(size_t)(r0 + a) * OC + o + b] = res[a][b] + (c->bias ? c->bias[o + b] : 0.0f);
        }
    }
}
static void matmul_forward(float* out, const float* inp, const float* w, const float* bias, int N, int C, int OC) {
    MMCtx c = { .out = out, .inp = inp, .w = w, .bias = bias, .N = N, .C = C, .OC = OC };
    parallel_for(N / 4, auto_chunk(N / 4), mm_fwd_range, &c);
}
static void mm_bwd_inp(void* p, int s, int e) {
    MMCtx* c = p; int C = c->C, OC = c->OC;
    for (int r = s; r < e; r++) {
        const float* d = c->dout + (size_t)r * OC; float* dx = c->dinp + (size_t)r * C;
        for (int o = 0; o < OC; o++) { const float* w = c->w + (size_t)o * C; float g = d[o]; for (int i = 0; i < C; i++) dx[i] += w[i] * g; }
    }
}
static void mm_bwd_w(void* p, int s, int e) { // s,e index blocks of 4 output channels
    MMCtx* c = p; int C = c->C, OC = c->OC, N = c->N;
    for (int blk = s; blk < e; blk++) {
        int o = blk * 4;
        float* d0 = c->dw + (size_t)o * C; float* d1 = d0 + C; float* d2 = d1 + C; float* d3 = d2 + C;
        float b0 = 0, b1 = 0, b2 = 0, b3 = 0;
        for (int r = 0; r < N; r++) {
            const float* g = c->dout + (size_t)r * OC + o; const float* x = c->inp + (size_t)r * C;
            float g0 = g[0], g1 = g[1], g2 = g[2], g3 = g[3];
            b0 += g0; b1 += g1; b2 += g2; b3 += g3;
            for (int i = 0; i < C; i++) { float xi = x[i]; d0[i] += xi * g0; d1[i] += xi * g1; d2[i] += xi * g2; d3[i] += xi * g3; }
        }
        if (c->db) { c->db[o] += b0; c->db[o + 1] += b1; c->db[o + 2] += b2; c->db[o + 3] += b3; }
    }
}
static void matmul_backward(float* dinp, float* dw, float* db, const float* dout, const float* inp, const float* w, int N, int C, int OC) {
    MMCtx c = { .dinp = dinp, .dw = dw, .db = db, .dout = dout, .inp = inp, .w = w, .N = N, .C = C, .OC = OC };
    parallel_for(N, auto_chunk(N), mm_bwd_inp, &c);
    parallel_for(OC / 4, auto_chunk(OC / 4), mm_bwd_w, &c);
}

typedef struct { float *out, *preatt, *att, *inp, *dinp, *dpreatt, *datt, *dout; int B, T, C, NH; } AttCtx;
static void att_fwd_range(void* p, int s, int e) {
    AttCtx* c = p; int T = c->T, C = c->C, NH = c->NH, C3 = 3 * C, hs = C / NH; float scale = 1.0f / sqrtf(hs);
    for (int idx = s; idx < e; idx++) {
        int b = idx / (NH * T), h = (idx / T) % NH, t = idx % T;
        const float* q = c->inp + (size_t)(b * T + t) * C3 + h * hs;
        float* pre = c->preatt + ((size_t)(b * NH + h) * T + t) * T; float* at = c->att + ((size_t)(b * NH + h) * T + t) * T;
        float mx = -1e30f;
        for (int t2 = 0; t2 <= t; t2++) {
            const float* k = c->inp + (size_t)(b * T + t2) * C3 + h * hs + C;
            float v = 0; for (int i = 0; i < hs; i++) v += q[i] * k[i]; v *= scale;
            pre[t2] = v; if (v > mx) mx = v;
        }
        float sum = 0;
        for (int t2 = 0; t2 <= t; t2++) { float ev = expf(pre[t2] - mx); at[t2] = ev; sum += ev; }
        float inv = 1.0f / sum;
        for (int t2 = 0; t2 < T; t2++) at[t2] = t2 <= t ? at[t2] * inv : 0.0f;
        float* o = c->out + (size_t)(b * T + t) * C + h * hs;
        for (int i = 0; i < hs; i++) o[i] = 0;
        for (int t2 = 0; t2 <= t; t2++) {
            const float* v = c->inp + (size_t)(b * T + t2) * C3 + h * hs + 2 * C; float a = at[t2];
            for (int i = 0; i < hs; i++) o[i] += a * v[i];
        }
    }
}
static void attention_forward(float* out, float* preatt, float* att, float* inp, int B, int T, int C, int NH) {
    AttCtx c = { .out = out, .preatt = preatt, .att = att, .inp = inp, .B = B, .T = T, .C = C, .NH = NH };
    int n = B * NH * T; parallel_for(n, auto_chunk(n), att_fwd_range, &c);
}
static void att_bwd_range(void* p, int s, int e) { // one item per (b,h): disjoint writes
    AttCtx* c = p; int T = c->T, C = c->C, NH = c->NH, C3 = 3 * C, hs = C / NH; float scale = 1.0f / sqrtf(hs);
    for (int idx = s; idx < e; idx++) {
        int b = idx / NH, h = idx % NH;
        for (int t = 0; t < T; t++) {
            const float* at = c->att + ((size_t)(b * NH + h) * T + t) * T;
            float* dat = c->datt + ((size_t)(b * NH + h) * T + t) * T;
            float* dpre = c->dpreatt + ((size_t)(b * NH + h) * T + t) * T;
            const float* q = c->inp + (size_t)(b * T + t) * C3 + h * hs; float* dq = c->dinp + (size_t)(b * T + t) * C3 + h * hs;
            const float* d = c->dout + (size_t)(b * T + t) * C + h * hs;
            for (int t2 = 0; t2 <= t; t2++) {
                const float* v = c->inp + (size_t)(b * T + t2) * C3 + h * hs + 2 * C; float* dv = c->dinp + (size_t)(b * T + t2) * C3 + h * hs + 2 * C;
                float acc = 0; float a = at[t2];
                for (int i = 0; i < hs; i++) { acc += v[i] * d[i]; dv[i] += a * d[i]; }
                dat[t2] += acc;
            }
            float dot = 0; for (int t2 = 0; t2 <= t; t2++) dot += at[t2] * dat[t2];
            for (int t2 = 0; t2 <= t; t2++) dpre[t2] += at[t2] * (dat[t2] - dot);
            for (int t2 = 0; t2 <= t; t2++) {
                const float* k = c->inp + (size_t)(b * T + t2) * C3 + h * hs + C; float* dk = c->dinp + (size_t)(b * T + t2) * C3 + h * hs + C;
                float g = dpre[t2] * scale;
                for (int i = 0; i < hs; i++) { dq[i] += k[i] * g; dk[i] += q[i] * g; }
            }
        }
    }
}
static void attention_backward(float* dinp, float* dpreatt, float* datt, float* dout, float* inp, float* att, int B, int T, int C, int NH) {
    AttCtx c = { .dinp = dinp, .dpreatt = dpreatt, .datt = datt, .dout = dout, .inp = inp, .att = att, .B = B, .T = T, .C = C, .NH = NH };
    parallel_for(B * NH, 1, att_bwd_range, &c);
}

#define GELU_K 0.7978845608f
typedef struct { float *out, *inp, *dinp, *dout; } GCtx;
static void gelu_fwd_range(void* p, int s, int e) {
    GCtx* c = p;
    for (int i = s; i < e; i++) { float x = c->inp[i]; c->out[i] = 0.5f * x * (1.0f + tanhf(GELU_K * (x + 0.044715f * x * x * x))); }
}
static void gelu_bwd_range(void* p, int s, int e) {
    GCtx* c = p;
    for (int i = s; i < e; i++) {
        float x = c->inp[i]; float th = tanhf(GELU_K * (x + 0.044715f * x * x * x));
        float sech2 = 1.0f - th * th; // safe under -ffast-math, unlike 1/cosh^2
        float g = 0.5f * (1.0f + th) + x * 0.5f * sech2 * GELU_K * (1.0f + 3.0f * 0.044715f * x * x);
        c->dinp[i] += g * c->dout[i];
    }
}
static void gelu_forward(float* out, float* inp, int N) { GCtx c = { .out = out, .inp = inp }; parallel_for(N, auto_chunk(N), gelu_fwd_range, &c); }
static void gelu_backward(float* dinp, float* inp, float* dout, int N) { GCtx c = { .dinp = dinp, .inp = inp, .dout = dout }; parallel_for(N, auto_chunk(N), gelu_bwd_range, &c); }

static void residual_forward(float* out, const float* a, const float* b, size_t N) { for (size_t i = 0; i < N; i++) out[i] = a[i] + b[i]; }
static void residual_backward(float* da, float* db, const float* dout, size_t N) { for (size_t i = 0; i < N; i++) { da[i] += dout[i]; db[i] += dout[i]; } }

static void softmax_forward(float* probs, const float* logits, int N, int V, int Vp) {
    for (int r = 0; r < N; r++) {
        const float* l = logits + (size_t)r * Vp; float* pr = probs + (size_t)r * Vp;
        float mx = -1e30f; for (int i = 0; i < V; i++) if (l[i] > mx) mx = l[i];
        float s = 0; for (int i = 0; i < V; i++) { pr[i] = expf(l[i] - mx); s += pr[i]; }
        for (int i = 0; i < V; i++) pr[i] /= s;
        for (int i = V; i < Vp; i++) pr[i] = 0;
    }
}

// ----------------------------------------------------------------------------
// model

typedef struct { int maxT, V, Vp, L, NH, C; } Config;
#define NPT 16
typedef struct { float *wte, *wpe, *ln1w, *ln1b, *qkvw, *qkvb, *attprojw, *attprojb, *ln2w, *ln2b, *fcw, *fcb, *fcprojw, *fcprojb, *lnfw, *lnfb; } Params;
#define NAT 23
typedef struct { float *encoded, *ln1, *ln1_mean, *ln1_rstd, *qkv, *atty, *preatt, *att, *attproj, *residual2, *ln2, *ln2_mean, *ln2_rstd,
                 *fch, *fch_gelu, *fcproj, *residual3, *lnf, *lnf_mean, *lnf_rstd, *logits, *probs, *losses; } Acts;

static void param_sizes(size_t* s, Config c) {
    size_t Vp = c.Vp, C = c.C, T = c.maxT, L = c.L;
    s[0] = Vp * C; s[1] = T * C; s[2] = L * C; s[3] = L * C; s[4] = L * 3 * C * C; s[5] = L * 3 * C; s[6] = L * C * C; s[7] = L * C;
    s[8] = L * C; s[9] = L * C; s[10] = L * 4 * C * C; s[11] = L * 4 * C; s[12] = L * C * 4 * C; s[13] = L * C; s[14] = C; s[15] = C;
}
static float* point_params(Params* p, size_t* s) {
    size_t n = 0; for (int i = 0; i < NPT; i++) n += s[i];
    float* mem = xmalloc(n * sizeof(float)); memset(mem, 0, n * sizeof(float));
    float** ptrs[] = { &p->wte, &p->wpe, &p->ln1w, &p->ln1b, &p->qkvw, &p->qkvb, &p->attprojw, &p->attprojb, &p->ln2w, &p->ln2b,
                       &p->fcw, &p->fcb, &p->fcprojw, &p->fcprojb, &p->lnfw, &p->lnfb };
    float* it = mem; for (int i = 0; i < NPT; i++) { *ptrs[i] = it; it += s[i]; }
    return mem;
}
static void act_sizes(size_t* s, Config c, int B, int T) {
    size_t C = c.C, NH = c.NH, L = c.L, Vp = c.Vp, Bs = B, Ts = T;
    s[0] = Bs * Ts * C; s[1] = L * Bs * Ts * C; s[2] = L * Bs * Ts; s[3] = L * Bs * Ts; s[4] = L * Bs * Ts * 3 * C; s[5] = L * Bs * Ts * C;
    s[6] = L * Bs * NH * Ts * Ts; s[7] = L * Bs * NH * Ts * Ts; s[8] = L * Bs * Ts * C; s[9] = L * Bs * Ts * C; s[10] = L * Bs * Ts * C;
    s[11] = L * Bs * Ts; s[12] = L * Bs * Ts; s[13] = L * Bs * Ts * 4 * C; s[14] = L * Bs * Ts * 4 * C; s[15] = L * Bs * Ts * C;
    s[16] = L * Bs * Ts * C; s[17] = Bs * Ts * C; s[18] = Bs * Ts; s[19] = Bs * Ts; s[20] = Bs * Ts * Vp; s[21] = Bs * Ts * Vp; s[22] = Bs * Ts;
}
static float* point_acts(Acts* a, size_t* s, size_t* total) {
    size_t n = 0; for (int i = 0; i < NAT; i++) n += s[i];
    float* mem = xmalloc(n * sizeof(float)); memset(mem, 0, n * sizeof(float));
    float** ptrs[] = { &a->encoded, &a->ln1, &a->ln1_mean, &a->ln1_rstd, &a->qkv, &a->atty, &a->preatt, &a->att, &a->attproj,
                       &a->residual2, &a->ln2, &a->ln2_mean, &a->ln2_rstd, &a->fch, &a->fch_gelu, &a->fcproj, &a->residual3,
                       &a->lnf, &a->lnf_mean, &a->lnf_rstd, &a->logits, &a->probs, &a->losses };
    float* it = mem; for (int i = 0; i < NAT; i++) { *ptrs[i] = it; it += s[i]; }
    *total = n; return mem;
}

typedef struct {
    Config cfg; Params p, g; size_t psz[NPT]; size_t nparams; float *pmem, *gmem, *m, *v;
    Acts a, ga; size_t asz[NAT]; size_t nacts; float *amem, *gamem; int B, T; int *inputs, *targets; float loss;
    float pdrop; int training; float *dm_enc, *dm_ap, *dm_fp; // residual-path dropout masks (0 or 1/(1-p))
} GPT;

static uint64_t rng = 1337;
static uint32_t rand_u32(uint64_t* s) { *s ^= *s >> 12; *s ^= *s << 25; *s ^= *s >> 27; return (*s * 0x2545F4914F6CDD1Dull) >> 32; }
static float rand_f32(uint64_t* s) { return (rand_u32(s) >> 8) / 16777216.0f; }
static float rand_normal(uint64_t* s) { float u1 = rand_f32(s) + 1e-7f, u2 = rand_f32(s); return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2); }

static void gpt_init(GPT* m, Config cfg, int B, int T) {
    m->cfg = cfg; param_sizes(m->psz, cfg);
    m->nparams = 0; for (int i = 0; i < NPT; i++) m->nparams += m->psz[i];
    m->pmem = point_params(&m->p, m->psz); m->gmem = point_params(&m->g, m->psz);
    m->m = calloc(m->nparams, sizeof(float)); m->v = calloc(m->nparams, sizeof(float));
    // nanoGPT init: N(0, 0.02); residual projections N(0, 0.02/sqrt(2L)); LN weight 1, biases 0
    float std = 0.02f, pstd = 0.02f / sqrtf(2.0f * cfg.L);
    for (size_t i = 0; i < (size_t)cfg.V * cfg.C; i++) m->p.wte[i] = std * rand_normal(&rng); // padded rows stay 0
    for (size_t i = 0; i < m->psz[1]; i++) m->p.wpe[i] = std * rand_normal(&rng);
    for (size_t i = 0; i < m->psz[4]; i++) m->p.qkvw[i] = std * rand_normal(&rng);
    for (size_t i = 0; i < m->psz[6]; i++) m->p.attprojw[i] = pstd * rand_normal(&rng);
    for (size_t i = 0; i < m->psz[10]; i++) m->p.fcw[i] = std * rand_normal(&rng);
    for (size_t i = 0; i < m->psz[12]; i++) m->p.fcprojw[i] = pstd * rand_normal(&rng);
    for (size_t i = 0; i < m->psz[2]; i++) { m->p.ln1w[i] = 1; m->p.ln2w[i] = 1; }
    for (int i = 0; i < cfg.C; i++) m->p.lnfw[i] = 1;
    m->B = B; m->T = T; act_sizes(m->asz, cfg, B, T);
    m->amem = point_acts(&m->a, m->asz, &m->nacts); m->gamem = point_acts(&m->ga, m->asz, &m->nacts);
    m->inputs = xmalloc(B * T * sizeof(int)); m->targets = xmalloc(B * T * sizeof(int));
    m->pdrop = 0; m->training = 0; m->dm_enc = m->dm_ap = m->dm_fp = NULL;
}
static uint64_t drop_rng = 99;
static void drop_apply(float* x, float* mask, size_t n, float p) { // forward: draw mask, scale in place
    float keep = 1.0f / (1.0f - p);
    for (size_t i = 0; i < n; i++) { mask[i] = rand_f32(&drop_rng) < p ? 0.0f : keep; x[i] *= mask[i]; }
}
static void drop_back(float* dx, const float* mask, size_t n) { for (size_t i = 0; i < n; i++) dx[i] *= mask[i]; }

// forward with B<=m->B, T<=m->T (smaller shapes reuse the same buffers; backward needs the full shape)
static void gpt_forward(GPT* m, const int* inputs, const int* targets, int B, int T) {
    int V = m->cfg.V, Vp = m->cfg.Vp, L = m->cfg.L, NH = m->cfg.NH, C = m->cfg.C, N = B * T;
    Params p = m->p; Acts a = m->a;
    memcpy(m->inputs, inputs, N * sizeof(int)); if (targets) memcpy(m->targets, targets, N * sizeof(int));
    encoder_forward(a.encoded, inputs, p.wte, p.wpe, B, T, C);
    int drop = m->training && m->pdrop > 0;
    if (drop) drop_apply(a.encoded, m->dm_enc, (size_t)N * C, m->pdrop);
    float* res;
    for (int l = 0; l < L; l++) {
        res = l == 0 ? a.encoded : a.residual3 + (size_t)(l - 1) * N * C;
        float *ln1 = a.ln1 + (size_t)l * N * C, *ln1m = a.ln1_mean + (size_t)l * N, *ln1r = a.ln1_rstd + (size_t)l * N, *qkv = a.qkv + (size_t)l * N * 3 * C,
              *atty = a.atty + (size_t)l * N * C, *pre = a.preatt + (size_t)l * B * NH * T * T, *att = a.att + (size_t)l * B * NH * T * T,
              *ap = a.attproj + (size_t)l * N * C, *r2 = a.residual2 + (size_t)l * N * C, *ln2 = a.ln2 + (size_t)l * N * C, *ln2m = a.ln2_mean + (size_t)l * N,
              *ln2r = a.ln2_rstd + (size_t)l * N, *fch = a.fch + (size_t)l * N * 4 * C, *fg = a.fch_gelu + (size_t)l * N * 4 * C,
              *fp = a.fcproj + (size_t)l * N * C, *r3 = a.residual3 + (size_t)l * N * C;
        layernorm_forward(ln1, ln1m, ln1r, res, p.ln1w + l * C, p.ln1b + l * C, N, C);
        matmul_forward(qkv, ln1, p.qkvw + (size_t)l * 3 * C * C, p.qkvb + l * 3 * C, N, C, 3 * C);
        attention_forward(atty, pre, att, qkv, B, T, C, NH);
        matmul_forward(ap, atty, p.attprojw + (size_t)l * C * C, p.attprojb + l * C, N, C, C);
        if (drop) drop_apply(ap, m->dm_ap + (size_t)l * N * C, (size_t)N * C, m->pdrop);
        residual_forward(r2, res, ap, (size_t)N * C);
        layernorm_forward(ln2, ln2m, ln2r, r2, p.ln2w + l * C, p.ln2b + l * C, N, C);
        matmul_forward(fch, ln2, p.fcw + (size_t)l * 4 * C * C, p.fcb + l * 4 * C, N, C, 4 * C);
        gelu_forward(fg, fch, N * 4 * C);
        matmul_forward(fp, fg, p.fcprojw + (size_t)l * C * 4 * C, p.fcprojb + l * C, N, 4 * C, C);
        if (drop) drop_apply(fp, m->dm_fp + (size_t)l * N * C, (size_t)N * C, m->pdrop);
        residual_forward(r3, r2, fp, (size_t)N * C);
    }
    res = a.residual3 + (size_t)(L - 1) * N * C;
    layernorm_forward(a.lnf, a.lnf_mean, a.lnf_rstd, res, p.lnfw, p.lnfb, N, C);
    matmul_forward(a.logits, a.lnf, p.wte, NULL, N, C, Vp);
    softmax_forward(a.probs, a.logits, N, V, Vp);
    if (targets) {
        double s = 0; for (int i = 0; i < N; i++) { float l = -logf(a.probs[(size_t)i * Vp + targets[i]] + 1e-30f); a.losses[i] = l; s += l; }
        m->loss = (float)(s / N);
    } else m->loss = -1;
}

static void gpt_backward(GPT* m) {
    int B = m->B, T = m->T, V = m->cfg.V, Vp = m->cfg.Vp, L = m->cfg.L, NH = m->cfg.NH, C = m->cfg.C, N = B * T;
    memset(m->gmem, 0, m->nparams * sizeof(float)); memset(m->gamem, 0, m->nacts * sizeof(float));
    Params p = m->p, g = m->g; Acts a = m->a, ga = m->ga;
    float dl = 1.0f / N;
    for (int r = 0; r < N; r++) {
        float* dlog = ga.logits + (size_t)r * Vp; const float* pr = a.probs + (size_t)r * Vp; int ix = m->targets[r];
        for (int i = 0; i < V; i++) dlog[i] = (pr[i] - (i == ix ? 1.0f : 0.0f)) * dl;
    }
    matmul_backward(ga.lnf, g.wte, NULL, ga.logits, a.lnf, p.wte, N, C, Vp);
    float* res = a.residual3 + (size_t)(L - 1) * N * C; float* dres = ga.residual3 + (size_t)(L - 1) * N * C;
    layernorm_backward(dres, g.lnfw, g.lnfb, ga.lnf, res, p.lnfw, a.lnf_mean, a.lnf_rstd, N, C);
    for (int l = L - 1; l >= 0; l--) {
        res = l == 0 ? a.encoded : a.residual3 + (size_t)(l - 1) * N * C;
        dres = l == 0 ? ga.encoded : ga.residual3 + (size_t)(l - 1) * N * C;
        size_t o1 = (size_t)l * N * C, o3 = o1 * 3, o4 = o1 * 4, oa = (size_t)l * B * NH * T * T, om = (size_t)l * N;
        residual_backward(ga.residual2 + o1, ga.fcproj + o1, ga.residual3 + o1, (size_t)N * C);
        if (m->pdrop > 0) drop_back(ga.fcproj + o1, m->dm_fp + o1, (size_t)N * C);
        matmul_backward(ga.fch_gelu + o4, g.fcprojw + (size_t)l * C * 4 * C, g.fcprojb + l * C, ga.fcproj + o1, a.fch_gelu + o4, p.fcprojw + (size_t)l * C * 4 * C, N, 4 * C, C);
        gelu_backward(ga.fch + o4, a.fch + o4, ga.fch_gelu + o4, N * 4 * C);
        matmul_backward(ga.ln2 + o1, g.fcw + (size_t)l * 4 * C * C, g.fcb + l * 4 * C, ga.fch + o4, a.ln2 + o1, p.fcw + (size_t)l * 4 * C * C, N, C, 4 * C);
        layernorm_backward(ga.residual2 + o1, g.ln2w + l * C, g.ln2b + l * C, ga.ln2 + o1, a.residual2 + o1, p.ln2w + l * C, a.ln2_mean + om, a.ln2_rstd + om, N, C);
        residual_backward(dres, ga.attproj + o1, ga.residual2 + o1, (size_t)N * C);
        if (m->pdrop > 0) drop_back(ga.attproj + o1, m->dm_ap + o1, (size_t)N * C);
        matmul_backward(ga.atty + o1, g.attprojw + (size_t)l * C * C, g.attprojb + l * C, ga.attproj + o1, a.atty + o1, p.attprojw + (size_t)l * C * C, N, C, C);
        attention_backward(ga.qkv + o3, ga.preatt + oa, ga.att + oa, ga.atty + o1, a.qkv + o3, a.att + oa, B, T, C, NH);
        matmul_backward(ga.ln1 + o1, g.qkvw + (size_t)l * 3 * C * C, g.qkvb + l * 3 * C, ga.qkv + o3, a.ln1 + o1, p.qkvw + (size_t)l * 3 * C * C, N, C, 3 * C);
        layernorm_backward(dres, g.ln1w + l * C, g.ln1b + l * C, ga.ln1 + o1, res, p.ln1w + l * C, a.ln1_mean + om, a.ln1_rstd + om, N, C);
    }
    if (m->pdrop > 0) drop_back(ga.encoded, m->dm_enc, (size_t)N * C);
    encoder_backward(g.wte, g.wpe, ga.encoded, m->inputs, B, T, C);
}

// AdamW, decay only 2D tensors (wte, wpe, qkvw, attprojw, fcw, fcprojw), grad clip by global norm
static float gpt_update(GPT* m, float lr, float b1, float b2, float eps, float wd, float clip, int t) {
    double ss = 0; for (size_t i = 0; i < m->nparams; i++) ss += (double)m->gmem[i] * m->gmem[i];
    float norm = (float)sqrt(ss), sc = (clip > 0 && norm > clip) ? clip / norm : 1.0f;
    float c1 = 1.0f - powf(b1, t), c2 = 1.0f - powf(b2, t);
    static const int decay[NPT] = { 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0 };
    size_t off = 0;
    for (int k = 0; k < NPT; k++) {
        float wdk = decay[k] ? wd : 0.0f;
        for (size_t j = 0; j < m->psz[k]; j++) {
            size_t i = off + j; float gr = m->gmem[i] * sc;
            float mm = b1 * m->m[i] + (1 - b1) * gr, vv = b2 * m->v[i] + (1 - b2) * gr * gr;
            m->m[i] = mm; m->v[i] = vv;
            m->pmem[i] -= lr * ((mm / c1) / (sqrtf(vv / c2) + eps) + wdk * m->pmem[i]);
        }
        off += m->psz[k];
    }
    return norm;
}

// ----------------------------------------------------------------------------
// data (character level), eval, sampling, checkpoints

static char vocab[256]; static int V_real = 0; static int stoi[256];
static int *train_ids, *val_ids; static size_t n_train, n_val;

static void load_data(const char* path) {
    FILE* f = fopen(path, "rb"); if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); size_t n = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char* buf = xmalloc(n); if (fread(buf, 1, n, f) != n) { fprintf(stderr, "read failed\n"); exit(1); } fclose(f);
    int seen[256] = { 0 }; for (size_t i = 0; i < n; i++) seen[buf[i]] = 1;
    for (int c = 0; c < 256; c++) { stoi[c] = -1; if (seen[c]) { stoi[c] = V_real; vocab[V_real++] = (char)c; } }
    n_train = (size_t)(n * 0.9); n_val = n - n_train; // nanoGPT split: first 90% train, last 10% val
    train_ids = xmalloc(n_train * sizeof(int)); val_ids = xmalloc(n_val * sizeof(int));
    for (size_t i = 0; i < n_train; i++) train_ids[i] = stoi[buf[i]];
    for (size_t i = 0; i < n_val; i++) val_ids[i] = stoi[buf[n_train + i]];
    free(buf);
}
static void get_batch(const int* ids, size_t n, int B, int T, int* x, int* y, uint64_t* r) {
    if (n <= (size_t)T + 1) { fprintf(stderr, "error: dataset too small for context length T=%d\n", T); exit(1); }
    for (int b = 0; b < B; b++) {
        size_t s = (size_t)(rand_f32(r) * (n - T - 1));
        for (int t = 0; t < T; t++) { x[b * T + t] = ids[s + t]; y[b * T + t] = ids[s + t + 1]; }
    }
}
static float estimate_loss(GPT* m, const int* ids, size_t n, int iters, uint64_t seed, int* x, int* y) {
    uint64_t r = seed; double s = 0;
    for (int i = 0; i < iters; i++) { get_batch(ids, n, m->B, m->T, x, y, &r); gpt_forward(m, x, y, m->B, m->T); s += m->loss; }
    return (float)(s / iters);
}
static void sample_text(GPT* m, int len, float temp, uint64_t seed, FILE* out) {
    int T = m->T; int* ctx = xmalloc(T * sizeof(int)); int n = 1; uint64_t r = seed;
    ctx[0] = stoi['\n'] >= 0 ? stoi['\n'] : 0;
    for (int k = 0; k < len; k++) {
        int Tc = n < T ? n : T; int Tp = (Tc + 3) / 4 * 4; // forward needs N % 4 == 0: right-pad, read position Tc-1 (causal)
        int* in = xmalloc(Tp * sizeof(int)); const int* src = ctx + (n - Tc);
        for (int t = 0; t < Tp; t++) in[t] = t < Tc ? src[t] : 0;
        gpt_forward(m, in, NULL, 1, Tp); free(in);
        float* lg = m->a.logits + (size_t)(Tc - 1) * m->cfg.Vp; float mx = -1e30f;
        for (int i = 0; i < m->cfg.V; i++) if (lg[i] > mx) mx = lg[i];
        float probs[256], s = 0; for (int i = 0; i < m->cfg.V; i++) { probs[i] = expf((lg[i] - mx) / temp); s += probs[i]; }
        float coin = rand_f32(&r) * s, cdf = 0; int nx = m->cfg.V - 1;
        for (int i = 0; i < m->cfg.V; i++) { cdf += probs[i]; if (coin < cdf) { nx = i; break; } }
        if (n == T) { memmove(ctx, ctx + 1, (T - 1) * sizeof(int)); ctx[T - 1] = nx; } else ctx[n++] = nx;
        fputc(vocab[nx], out);
    }
    fputc('\n', out); fflush(out); free(ctx);
}
static void save_ckpt(GPT* m, const char* path, int step) {
    FILE* f = fopen(path, "wb"); if (!f) return;
    int hdr[16] = { 20261103, 1, m->cfg.maxT, m->cfg.V, m->cfg.Vp, m->cfg.L, m->cfg.NH, m->cfg.C, step };
    fwrite(hdr, sizeof(int), 16, f); fwrite(vocab, 1, 256, f); fwrite(m->pmem, sizeof(float), m->nparams, f); fclose(f);
}
static int read_int_file(const char* p) { FILE* f = fopen(p, "r"); if (!f) return -99999; int v = -99999; if (fscanf(f, "%d", &v) != 1) v = -99999; fclose(f); return v; }

// ----------------------------------------------------------------------------

int main(int argc, char** argv) {
    const char* data = "input.txt"; const char* outdir = "."; const char* tag = "run";
    int L = 4, NH = 4, C = 128, T = 64, B = 12, iters = 2000, warm = 100, decay_iters = -1, eval_int = 250, eval_iters = 20, threads = 8, seed = 1337;
    int save_every = 0, gen_len = 300, log_int = 10, maxtemp10 = 450, sample_every = 0; float pdrop = 0.0f;
    float lr = 1e-3f, min_lr = 1e-4f, b1 = 0.9f, b2 = 0.99f, wd = 0.1f, clip = 1.0f; double max_minutes = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        const char* k = argv[i]; const char* v = argv[i + 1];
        if (!strcmp(k, "-data")) data = v; else if (!strcmp(k, "-out")) outdir = v; else if (!strcmp(k, "-tag")) tag = v;
        else if (!strcmp(k, "-L")) L = atoi(v); else if (!strcmp(k, "-H")) NH = atoi(v); else if (!strcmp(k, "-C")) C = atoi(v);
        else if (!strcmp(k, "-T")) T = atoi(v); else if (!strcmp(k, "-B")) B = atoi(v); else if (!strcmp(k, "-iters")) iters = atoi(v);
        else if (!strcmp(k, "-warm")) warm = atoi(v); else if (!strcmp(k, "-decay")) decay_iters = atoi(v);
        else if (!strcmp(k, "-ei")) eval_int = atoi(v); else if (!strcmp(k, "-eb")) eval_iters = atoi(v);
        else if (!strcmp(k, "-threads")) threads = atoi(v); else if (!strcmp(k, "-seed")) seed = atoi(v);
        else if (!strcmp(k, "-save")) save_every = atoi(v); else if (!strcmp(k, "-gen")) gen_len = atoi(v); else if (!strcmp(k, "-log")) log_int = atoi(v);
        else if (!strcmp(k, "-lr")) lr = atof(v); else if (!strcmp(k, "-minlr")) min_lr = atof(v); else if (!strcmp(k, "-b2")) b2 = atof(v);
        else if (!strcmp(k, "-wd")) wd = atof(v); else if (!strcmp(k, "-clip")) clip = atof(v); else if (!strcmp(k, "-minutes")) max_minutes = atof(v);
        else if (!strcmp(k, "-maxtemp")) maxtemp10 = (int)(atof(v) * 10);
        else if (!strcmp(k, "-dropout")) pdrop = atof(v);
        else if (!strcmp(k, "-sample_every")) sample_every = atoi(v);
        else { fprintf(stderr, "unknown arg %s\n", k); return 1; }
    }
    if (decay_iters < 0) decay_iters = iters;
    if ((B * T) % 4 || C % 4 || C % NH) { fprintf(stderr, "need B*T%%4==0, C%%4==0, C%%H==0\n"); return 1; }
    rng = (uint64_t)seed * 2654435761u + 1;
    drop_rng = (uint64_t)seed + 13;
    pool_init(threads);
    load_data(data);
    Config cfg = { .maxT = T, .V = V_real, .Vp = (V_real + 3) / 4 * 4, .L = L, .NH = NH, .C = C };
    GPT m; gpt_init(&m, cfg, B, T);
    if (pdrop > 0) { m.pdrop = pdrop; m.dm_enc = xmalloc((size_t)B * T * C * sizeof(float));
        m.dm_ap = xmalloc((size_t)L * B * T * C * sizeof(float)); m.dm_fp = xmalloc((size_t)L * B * T * C * sizeof(float)); }
    char path[512]; snprintf(path, sizeof path, "%s/%s_log.txt", outdir, tag); FILE* lg = fopen(path, "w");
    size_t nonemb = m.nparams - m.psz[0] - m.psz[1];
    printf("[phone_gpt] data=%s chars train=%zu val=%zu vocab=%d | L=%d H=%d C=%d T=%d B=%d | params=%zu (non-embedding %zu) | threads=%d\n",
           data, n_train, n_val, V_real, L, NH, C, T, B, m.nparams, nonemb, threads);
    fprintf(lg, "# params=%zu nonemb=%zu L=%d H=%d C=%d T=%d B=%d iters=%d dropout=%g lr=%g minlr=%g warm=%d decay=%d wd=%g b2=%g clip=%g threads=%d seed=%d\n",
            m.nparams, nonemb, L, NH, C, T, B, iters, pdrop, lr, min_lr, warm, decay_iters, wd, b2, clip, threads, seed);
    fprintf(lg, "kind,step,elapsed_s,train_loss,val_loss,lr,grad_norm,ms_per_step,tok_per_s,batt_temp_c,batt_pct,current_ma\n"); fflush(lg);
    int* x = xmalloc(B * T * sizeof(int)); int* y = xmalloc(B * T * sizeof(int));
    uint64_t data_rng = (uint64_t)seed + 7;
    double t0 = now_s(), win_t = now_s(); int win_steps = 0; float win_loss = 0;
    for (int step = 0; step <= iters; step++) {
        int last = step == iters || (max_minutes > 0 && (now_s() - t0) / 60.0 >= max_minutes);
        if (step % eval_int == 0 || last) {
            double te = now_s();
            float tl = estimate_loss(&m, train_ids, n_train, eval_iters, 4242, x, y);
            float vl = estimate_loss(&m, val_ids, n_val, eval_iters, 4243, x, y);
            int temp = read_int_file("/sys/class/power_supply/battery/temp"), cap = read_int_file("/sys/class/power_supply/battery/capacity");
            int cur = read_int_file("/sys/class/power_supply/battery/current_now");
            printf("EVAL step %d | train %.4f | val %.4f | %.1f min | eval %.1fs | batt %.1fC %d%% %dmA\n", step, tl, vl, (now_s() - t0) / 60, now_s() - te, temp / 10.0, cap, cur);
            fprintf(lg, "eval,%d,%.1f,%.5f,%.5f,,,,,%.1f,%d,%d\n", step, now_s() - t0, tl, vl, temp / 10.0, cap, cur); fflush(lg);
            if (sample_every > 0 && step % sample_every == 0) { // live samples for the on-phone page
                snprintf(path, sizeof path, "%s/%s_samples.txt", outdir, tag); FILE* sf2 = fopen(path, "a");
                if (sf2) { fprintf(sf2, "### step %d | held-out %.4f\n", step, vl); sample_text(&m, 240, 0.8f, 1000 + step, sf2); fclose(sf2); }
            }
            if (save_every > 0 && (step % save_every == 0 || last)) { // step 0 too: the random start can be shown
                snprintf(path, sizeof path, "%s/%s_step%06d.bin", outdir, tag, step); save_ckpt(&m, path, step);
            }
            fflush(stdout);
        }
        if (last) break;
        // thermal guard: pause while the battery is hotter than maxtemp
        int temp = read_int_file("/sys/class/power_supply/battery/temp");
        while (temp != -99999 && temp > maxtemp10) {
            printf("THERMAL pause at step %d: %.1fC\n", step, temp / 10.0); fflush(stdout);
            fprintf(lg, "pause,%d,%.1f,,,,,,,%.1f,,\n", step, now_s() - t0, temp / 10.0); fflush(lg);
            sleep(30); temp = read_int_file("/sys/class/power_supply/battery/temp");
            if (temp <= maxtemp10 - 30) break;
        }
        float cur_lr;
        if (step < warm) cur_lr = lr * (step + 1) / (float)(warm + 1);
        else if (step > decay_iters) cur_lr = min_lr;
        else { float r = (float)(step - warm) / (float)(decay_iters - warm); cur_lr = min_lr + 0.5f * (1.0f + cosf(3.14159265f * r)) * (lr - min_lr); }
        double ts = now_s();
        get_batch(train_ids, n_train, B, T, x, y, &data_rng);
        m.training = 1; gpt_forward(&m, x, y, B, T); m.training = 0;
        gpt_backward(&m);
        float gn = gpt_update(&m, cur_lr, b1, b2, 1e-8f, wd, clip, step + 1);
        double ms = (now_s() - ts) * 1000;
        win_steps++; win_loss += m.loss;
        if (step % log_int == 0) {
            double el = now_s() - win_t; double tps = win_steps * (double)B * T / el;
            int temp2 = read_int_file("/sys/class/power_supply/battery/temp"), cap = read_int_file("/sys/class/power_supply/battery/capacity");
            int cur = read_int_file("/sys/class/power_supply/battery/current_now");
            printf("step %5d | loss %.4f | lr %.2e | norm %.2f | %.0f ms | %.0f tok/s | %.1fC\n", step, win_loss / win_steps, cur_lr, gn, ms, tps, temp2 / 10.0);
            fprintf(lg, "train,%d,%.1f,%.5f,,%.6g,%.4f,%.1f,%.1f,%.1f,%d,%d\n", step, now_s() - t0, win_loss / win_steps, cur_lr, gn, ms, tps, temp2 / 10.0, cap, cur);
            fflush(lg); fflush(stdout); win_t = now_s(); win_steps = 0; win_loss = 0;
        }
    }
    printf("TOTAL %.1f min\n--- sample (temperature 0.8) ---\n", (now_s() - t0) / 60);
    snprintf(path, sizeof path, "%s/%s_sample.txt", outdir, tag); FILE* sf = fopen(path, "w");
    sample_text(&m, gen_len, 0.8f, 99, stdout); if (sf) { sample_text(&m, gen_len, 0.8f, 99, sf); fclose(sf); }
    snprintf(path, sizeof path, "%s/%s_final.bin", outdir, tag); save_ckpt(&m, path, iters);
    fclose(lg);
    return 0;
}
