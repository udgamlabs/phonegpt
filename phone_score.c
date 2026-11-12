// phone_score.c: score a phone_gpt checkpoint on the WHOLE held-out split, with no sampling.
// The trainer's held-out number uses a fixed small draw of windows (-eb batches of -B). This scores every window start
// (stride 1), so the answer is the exact mean a random draw of that context length is estimating, and the spread of the
// per-window means tells how noisy a small draw is. Shares the trainer's code, so the maths cannot drift from it.
//   clang -O3 -ffast-math -o phone_score phone_score.c -lpthread -lm
//   ./phone_score <ckpt.bin> <input.txt> <out_prefix> [threads=4] [stride=1] [eb=16]
// Writes <out_prefix>_windows.f32 (one mean loss per window), <out_prefix>_positions.csv (mean loss by position in the
// window) and <out_prefix>_summary.txt.
#define main phone_gpt_main
#include "phone_gpt.c"
#undef main

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: phone_score ckpt input.txt out_prefix [threads] [stride] [eb]\n"); return 1; }
    int threads = argc > 4 ? atoi(argv[4]) : 4, stride = argc > 5 ? atoi(argv[5]) : 1, eb = argc > 6 ? atoi(argv[6]) : 16, B = 16;
    FILE* f = fopen(argv[1], "rb"); if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    int hdr[16]; char ck_vocab[256];
    if (fread(hdr, sizeof(int), 16, f) != 16 || hdr[0] != 20261103 || fread(ck_vocab, 1, 256, f) != 256) { fprintf(stderr, "bad checkpoint\n"); return 1; }
    pool_init(threads);
    load_data(argv[2]);
    if (hdr[3] != V_real || memcmp(ck_vocab, vocab, V_real)) { fprintf(stderr, "vocabulary differs from the checkpoint\n"); return 1; }
    if (hdr[2] <= 0 || hdr[2] > (1 << 20)) { fprintf(stderr, "error: checkpoint context length %d out of range\n", hdr[2]); return 1; }
    if (hdr[4] < hdr[3] || hdr[4] > 1024) { fprintf(stderr, "error: checkpoint padded vocab size %d out of range\n", hdr[4]); return 1; }
    if (hdr[5] <= 0 || hdr[5] > 1000) { fprintf(stderr, "error: checkpoint layer count %d out of range\n", hdr[5]); return 1; }
    if (hdr[6] <= 0 || hdr[6] > 1024) { fprintf(stderr, "error: checkpoint head count %d out of range\n", hdr[6]); return 1; }
    if (hdr[7] <= 0 || hdr[7] > (1 << 20)) { fprintf(stderr, "error: checkpoint channel count %d out of range\n", hdr[7]); return 1; }
    int T = hdr[2];
    Config cfg = { .maxT = T, .V = hdr[3], .Vp = hdr[4], .L = hdr[5], .NH = hdr[6], .C = hdr[7] };
    GPT m; gpt_init(&m, cfg, B, T);
    if (fread(m.pmem, sizeof(float), m.nparams, f) != m.nparams) { fprintf(stderr, "short checkpoint\n"); return 1; }
    fclose(f);
    int* x = xmalloc(B * T * sizeof(int)); int* y = xmalloc(B * T * sizeof(int));

    // parity: the trainer's own fixed draws must come back
    float tl = estimate_loss(&m, train_ids, n_train, eb, 4242, x, y), vl = estimate_loss(&m, val_ids, n_val, eb, 4243, x, y);

    if (n_val <= (size_t)T + 1) { fprintf(stderr, "error: held-out split too small for context length T=%d\n", T); return 1; }
    size_t nw = (n_val - T - 1) / stride + 1; // starts 0, stride, ...; the last target is val_ids[n_val - 1]
    float* win = xmalloc(nw * sizeof(float)); double* pos = calloc(T, sizeof(double)); double total = 0;
    double t0 = now_s(); size_t done = 0;
    while (done < nw) {
        int b = nw - done < (size_t)B ? (int)(nw - done) : B;
        for (int i = 0; i < b; i++) { size_t s = (done + i) * stride; for (int t = 0; t < T; t++) { x[i * T + t] = val_ids[s + t]; y[i * T + t] = val_ids[s + t + 1]; } }
        gpt_forward(&m, x, y, b, T);
        for (int i = 0; i < b; i++) {
            double w = 0; for (int t = 0; t < T; t++) { float l = m.a.losses[i * T + t]; w += l; pos[t] += l; }
            win[done + i] = (float)(w / T); total += w / T;
        }
        done += b;
        if ((done / B) % 500 == 0) { printf("%zu / %zu windows, %.1f min, running mean %.5f\n", done, nw, (now_s() - t0) / 60, total / done); fflush(stdout); }
    }
    char path[1024];
    snprintf(path, sizeof path, "%s_windows.f32", argv[3]); f = fopen(path, "wb"); fwrite(win, sizeof(float), nw, f); fclose(f);
    snprintf(path, sizeof path, "%s_positions.csv", argv[3]); f = fopen(path, "w"); fprintf(f, "position,mean_loss\n");
    for (int t = 0; t < T; t++) fprintf(f, "%d,%.5f\n", t, pos[t] / nw); fclose(f);
    snprintf(path, sizeof path, "%s_summary.txt", argv[3]); f = fopen(path, "w");
    for (int k = 0; k < 2; k++) {
        FILE* o = k ? stdout : f;
        fprintf(o, "checkpoint step %d | L=%d H=%d C=%d T=%d | params %zu | held-out chars %zu\n", hdr[8], cfg.L, cfg.NH, cfg.C, T, m.nparams, n_val);
        fprintf(o, "trainer's fixed draws (%d x %d windows): train %.5f | held-out %.5f\n", eb, B, tl, vl);
        fprintf(o, "whole held-out split: %zu windows, stride %d\n", nw, stride);
        fprintf(o, "mean over every position of every window: %.5f\n", total / nw);
        fprintf(o, "last position only (full %d-char context): %.5f\n", T - 1, pos[T - 1] / nw);
        fprintf(o, "time %.1f min, %d threads\n", (now_s() - t0) / 60, threads);
    }
    fclose(f);
    return 0;
}
