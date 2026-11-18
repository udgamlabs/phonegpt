// phone_sample.c: write text from a phone_gpt checkpoint. No training text is needed; the checkpoint holds its alphabet.
//   zig cc -O3 -ffast-math -o phone_sample phone_sample.c -lpthread -lm
//   ./phone_sample <ckpt.bin> [chars=500] [seed=1] [temperature=0.8]
#define main phone_gpt_main
#include "phone_gpt.c"
#undef main

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: phone_sample ckpt [chars] [seed] [temperature]\n"); return 1; }
    int len = argc > 2 ? atoi(argv[2]) : 500, seed = argc > 3 ? atoi(argv[3]) : 1; float temp = argc > 4 ? atof(argv[4]) : 0.8f;
    FILE* f = fopen(argv[1], "rb"); if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    int hdr[16];
    if (fread(hdr, sizeof(int), 16, f) != 16 || hdr[0] != 20261103 || fread(vocab, 1, 256, f) != 256) { fprintf(stderr, "bad checkpoint\n"); return 1; }
    if (hdr[3] < 0 || hdr[3] > 256) { fprintf(stderr, "error: checkpoint vocab size %d exceeds max 256\n", hdr[3]); return 1; }
    if (hdr[2] <= 0 || hdr[2] > (1 << 20)) { fprintf(stderr, "error: checkpoint context length %d out of range\n", hdr[2]); return 1; }
    if (hdr[4] < hdr[3] || hdr[4] > 1024) { fprintf(stderr, "error: checkpoint padded vocab size %d out of range\n", hdr[4]); return 1; }
    if (hdr[5] <= 0 || hdr[5] > 1000) { fprintf(stderr, "error: checkpoint layer count %d out of range\n", hdr[5]); return 1; }
    if (hdr[6] <= 0 || hdr[6] > 1024) { fprintf(stderr, "error: checkpoint head count %d out of range\n", hdr[6]); return 1; }
    if (hdr[7] <= 0 || hdr[7] > (1 << 20)) { fprintf(stderr, "error: checkpoint channel count %d out of range\n", hdr[7]); return 1; }
    V_real = hdr[3];
    for (int c = 0; c < 256; c++) stoi[c] = -1;
    for (int i = 0; i < V_real; i++) stoi[(unsigned char)vocab[i]] = i;
    pool_init(2);
    Config cfg = { .maxT = hdr[2], .V = hdr[3], .Vp = hdr[4], .L = hdr[5], .NH = hdr[6], .C = hdr[7] };
    GPT m; gpt_init(&m, cfg, 16, hdr[2]);
    if (fread(m.pmem, sizeof(float), m.nparams, f) != m.nparams) { fprintf(stderr, "short checkpoint\n"); return 1; }
    fclose(f);
    sample_text(&m, len, temp, seed, stdout);
    return 0;
}
