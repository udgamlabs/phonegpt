# PhoneGPT, by Udgam Labs

Train a small language model on an Android phone's own CPU, from random weights, with no root and no app.

**Links:** [the write-up](https://udgamlabs.com/blog/phonegpt) · [the trained model on Hugging Face](https://huggingface.co/udgamlabs/PhoneGPT) · [the stream recording](https://www.youtube.com/watch?v=gXQFqCBB91U)

We ran this live on a 2021 Galaxy F62 against the 1.4697 in nanoGPT's README. It missed by 0.0047: best 1.4663 at one
checkpoint, not two in a row. Story, results and limits: https://udgamlabs.com/blog/phonegpt

It was a fun experiment, not a useful model. It writes Shakespeare-shaped nonsense. PhoneGPT is only a project name, with
no link to the offline-AI iPhone app of the same name. We have only tested on the F62, from a Mac.

## Run it

You need a phone with USB debugging on, a USB cable, and a computer with `adb` and [Zig](https://ziglang.org/download/).

```bash
curl -sSL -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
zig cc -target aarch64-linux-musl -O3 -ffast-math -static -o phone_gpt phone_gpt.c -lpthread -lm
adb shell mkdir -p /data/local/tmp/phonegpt
adb push phone_gpt input.txt /data/local/tmp/phonegpt/
adb shell chmod +x /data/local/tmp/phonegpt/phone_gpt
adb shell svc power stayon usb
adb shell 'cd /data/local/tmp/phonegpt && (nohup taskset c0 ./phone_gpt -threads 2 -L 6 -H 4 -C 128 -T 128 -B 16 -iters 18000 -warm 100 -lr 2e-3 -minlr 2e-4 -dropout 0.1 -ei 250 -eb 16 -log 25 -save 1000 -sample_every 250 -minutes 30 -maxtemp 45 -gen 1000 -tag run </dev/null > out.txt 2>&1 &)'
```

This trial stops after 30 minutes. For the full run, use `-minutes 510`; 18,000 steps took about 8 hours on the F62.
To replay our live run, use `-iters 24000 -decay 18000 -minutes 680 -seed 20261105`. A fresh build on the same phone matched
the live run's log at every logged point over the first 277 steps, which is as far as we checked. That match predates the
fix that seeds `drop_rng` from `-seed` instead of a constant: dropout now varies with the seed as intended, so a replay on
the current build will diverge from the published log once dropout actually drops anything, even with the same `-seed`.

`taskset c0` pins to cpu6 and cpu7, the F62's fastest cores, and `-threads` matches that count. Check yours first, and
change the hex mask to fit (`80` is cpu7 alone, `f0` is cpu4 to cpu7):

```bash
adb shell 'for c in /sys/devices/system/cpu/cpu[0-9]*; do echo "$(basename $c) $(cat $c/cpufreq/cpuinfo_max_freq)"; done'
```

Watch it, and stop it early if you like:

```bash
adb shell tail -n 30 /data/local/tmp/phonegpt/run_samples.txt
adb shell tail -n 3 /data/local/tmp/phonegpt/run_log.txt
adb shell pkill phone_gpt
```

## Score it

The held-out number in the trainer's log is a quick check on a small sample and can read low. For a number you can
compare, score every window on your computer:

```bash
adb pull /data/local/tmp/phonegpt/run_final.bin
zig cc -O3 -ffast-math -o phone_score phone_score.c -lpthread -lm
./phone_score run_final.bin input.txt score 4 1 16
cat score_summary.txt
adb shell svc power stayon false
```

`run_final.bin` appears when a run ends by itself, and its last text is in `run_sample.txt`. If you stop early, score the
newest `run_step*.bin` instead. After a run stopped by `-minutes`, the summary's first line shows the planned last step, not
the step it really reached; the score is still that checkpoint's.

`results/scores.csv` holds the live run's score at each checkpoint (step, score).

## Write text from the trained model

The checkpoints from our live run are at https://huggingface.co/udgamlabs/PhoneGPT. This needs no phone and no training text:

```bash
curl -LO https://huggingface.co/udgamlabs/PhoneGPT/resolve/main/checkpoints/live_step018000.bin
zig cc -O3 -ffast-math -o phone_sample phone_sample.c -lpthread -lm
./phone_sample live_step018000.bin 500
```

## Credit

`phone_gpt.c` is adapted from [llm.c](https://github.com/karpathy/llm.c) by Andrej Karpathy (MIT). The target comes from
his [nanoGPT](https://github.com/karpathy/nanoGPT) and the text from his [char-rnn](https://github.com/karpathy/char-rnn).
He isn't involved in this.

## About

Udgam Labs · AI research from first principles
https://udgamlabs.com
