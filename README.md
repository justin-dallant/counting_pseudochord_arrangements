# counting_pseudochord_arrangements

This is the supporting code for the preprint "Justin Dallant, Improved Lower Bound on the Number of Pseudoline Arrangements" (https://arxiv.org/abs/2402.13923).

Compile with:

```
cargo build --release
```

Usage:

```
count_arrangements [OPTIONS] --chords <CHORDS> 
 Options:
  -c, --chords <CHORDS>            Chords as pairs of integers between quotation marks. E.g. "(0,3), (1,4), (2,5)"
  -t, --timeout <TIMEOUT>          (Optional) Max time, in seconds, before atempting to abort the computation
  -n, --num-threads <NUM_THREADS>  (Optional) Number of threads to spawn (default: 1)
  -h, --help                       Print help
  -V, --version                    Print version
```

Example (counts the number of pseudoline arrangements of order 8):

```
count_arrangements --chords '(0,8), (1,9), (2,10), (3,11), (4,12), (5,13), (6,14), (7,15)' --num-threads 4 --timeout 120
```
