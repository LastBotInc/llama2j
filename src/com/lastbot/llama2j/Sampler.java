package com.lastbot.llama2j;

import it.unimi.dsi.util.XoRoShiRo128PlusRandom;

import java.util.Arrays;
import java.util.Random;

public class Sampler {
    private final Random random;
    private final ProbIndex[] probIndex; // buffer used in top-p sampling, CPU only

    private float random_f32() { // random float32 in [0,1)
        return random.nextFloat();
    }

    public Sampler(long rngSeed, int vocab_size) {
        this.random = new XoRoShiRo128PlusRandom(rngSeed);

        probIndex = new ProbIndex[vocab_size];
        for (int i = 0; i < vocab_size; i++) {
            probIndex[i] = new ProbIndex();
        }
    }

    public int sample(float[] logits, int n) {
        // sample index from probabilities, they must sum to 1
        float r = random_f32();
        float cdf = 0.0f;
        for (int i = 0; i < n; i++) {
            cdf += logits[i];
            if (r < cdf) {
                return i;
            }
        }
        return n - 1; // in case of rounding errors
    }

    public int argmax(float[] logits, int n) {
        // return argmax of logits in elements 0..n
        int max_i = 0;
        float max_p = logits[0];
        for (int i = 1; i < n; i++) {
            if (logits[i] > max_p) {
                max_i = i;
                max_p = logits[i];
            }
        }
        return max_i;
    }


    // struct used when sorting probabilities during top-p sampling, CPU only
    public static class ProbIndex implements Comparable<ProbIndex> {
        public float prob;
        public int index;

        @Override
        public int compareTo(ProbIndex o) {
            if (prob > o.prob) {
                return -1;
            } else if (prob < o.prob) {
                return 1;
            } else {
                return 0;
            }
        }
    }

    public int sample_topp(float[] logits, int n, float topp) {
        // top-p sampling (or "nucleus sampling") samples from the smallest set of
        // tokens that exceed probability topp. This way we never sample tokens that
        // have very low logits and are less likely to go "off the rails".

        int n0 = 0;
        // quicksort indices in descending order of logits
        // values smaller than (1 - topp) / (n - 1) cannot be part of the result
        // so for efficiency we crop these out as candidates before sorting
        float cutoff = (1.0f - topp) / (n - 1);

        for (int i = 0; i < n; i++) {
            if (logits[i] >= cutoff) {
                probIndex[n0].index = i;
                probIndex[n0].prob = logits[i];
                n0++;
            }
        }
        Arrays.sort(probIndex);

        // truncate the list where cumulative probability exceeds topp
        float cumulative_prob = 0.0f;
        int last_idx = n0 - 1; // in case of rounding errors consider all elements
        for (int i = 0; i < n0; i++) {
            cumulative_prob += probIndex[i].prob;
            if (cumulative_prob > topp) {
                last_idx = i;
                break; // we've exceeded topp by including last_idx
            }
        }

        // sample from the truncated list
        float r = random_f32() * cumulative_prob;
        float cdf = 0.0f;
        for (int i = 0; i <= last_idx; i++) {
            cdf += probIndex[i].prob;
            if (r < cdf) {
                return probIndex[i].index;
            }
        }
        return probIndex[last_idx].index; // in case of rounding errors
    }
}
