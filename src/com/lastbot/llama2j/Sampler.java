package com.lastbot.llama2j;

import it.unimi.dsi.util.XoRoShiRo128PlusRandom;

import java.util.Arrays;
import java.util.Random;

public class Sampler {
    private final Random random;

    private float random_f32() { // random float32 in [0,1)
        return random.nextFloat();
    }

    public Sampler(long rngSeed) {
        this.random = new XoRoShiRo128PlusRandom(rngSeed);
    }

    public int sample(RunState state, int n) {
        float[] probabilities = state.logits;
        // sample index from probabilities, they must sum to 1
        float r = random_f32();
        float cdf = 0.0f;
        for (int i = 0; i < n; i++) {
            cdf += probabilities[i];
            if (r < cdf) {
                return i;
            }
        }
        return n - 1; // in case of rounding errors
    }

    public int argmax(float[] v, int n) {
        // return argmax of v in elements 0..n
        int max_i = 0;
        float max_p = v[0];
        for (int i = 1; i < n; i++) {
            if (v[i] > max_p) {
                max_i = i;
                max_p = v[i];
            }
        }
        return max_i;
    }

    public int sample_topp(RunState state, int n, float topp) {
        // top-p sampling (or "nucleus sampling") samples from the smallest set of
        // tokens that exceed probability topp. This way we never sample tokens that
        // have very low probabilities and are less likely to go "off the rails".

        float[] probabilities = state.logits;

        int n0 = 0;
        // quicksort indices in descending order of probabilities
        // values smaller than (1 - topp) / (n - 1) cannot be part of the result
        // so for efficiency we crop these out as candidates before sorting
        float cutoff = (1.0f - topp) / (n - 1);

        for (int i = 0; i < n; i++) {
            if (probabilities[i] >= cutoff) {
                state.probIndex[n0].index = i;
                state.probIndex[n0].prob = probabilities[i];
                n0++;
            }
        }
        Arrays.sort(state.probIndex);

        // truncate the list where cumulative probability exceeds topp
        float cumulative_prob = 0.0f;
        int last_idx = n0 - 1; // in case of rounding errors consider all elements
        for (int i = 0; i < n0; i++) {
            cumulative_prob += state.probIndex[i].prob;
            if (cumulative_prob > topp) {
                last_idx = i;
                break; // we've exceeded topp by including last_idx
            }
        }

        // sample from the truncated list
        float r = random_f32() * cumulative_prob;
        float cdf = 0.0f;
        for (int i = 0; i <= last_idx; i++) {
            cdf += state.probIndex[i].prob;
            if (r < cdf) {
                return state.probIndex[i].index;
            }
        }
        return state.probIndex[last_idx].index; // in case of rounding errors
    }
}
