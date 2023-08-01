package com.lastbot.llama2j;

/*
Inference for Llama-2 Transformer model in pure Java.

Adapted from: :https://github.com/karpathy/llama2.c

Commit: ce05cc2

*/

import it.unimi.dsi.util.XoRoShiRo128PlusRandom;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.concurrent.CountDownLatch;

import static java.lang.Math.abs;

public class Run {
    private static final int THREAD_COUNT = 24;

    private static final String MODELS_DIRECTORY = "models";
    private static final String TOKENIZER_FILE = "tokenizer.bin";

    private static int maxCallocSize = 0;

    private static XoRoShiRo128PlusRandom random = new XoRoShiRo128PlusRandom();

    private static float[] calloc(long elements) {
        if (elements <= Integer.MAX_VALUE) {
            int size = (int) elements;
            if (size > maxCallocSize) {
                maxCallocSize = size;
            }
            return new float[size];
        }

        double ratio = (double) elements / (double) Integer.MAX_VALUE;
        String s = "Tried to allocate " + String.format("%,d", elements) +
                " elements, which is " + String.format("%,.1f", ratio) +
                " times more than supported";
        LLogger.error(s);
        System.exit(1);
        return null;
    }

    private static RunState mallocRunState(Config p) {
        RunState s = new RunState();
        s.x = calloc(p.dim);
        s.xb = calloc(p.dim);
        s.xb2 = calloc(p.dim);
        s.hb = calloc(p.hidden_dim);
        s.hb2 = calloc(p.hidden_dim);
        s.q = calloc(p.dim);
        s.k = calloc(p.dim);
        s.v = calloc(p.dim);
        s.att = calloc(p.n_heads * p.seq_len);
        s.logits = calloc(p.vocab_size);
        s.key_cache = calloc(p.n_layers * p.seq_len * p.dim);
        s.value_cache = calloc(p.n_layers * p.seq_len * p.dim);
        return s;
    }

    private static void free_run_state(RunState s) {
//        free(s -> x);
//        free(s -> xb);
//        free(s -> xb2);
//        free(s -> hb);
//        free(s -> hb2);
//        free(s -> q);
//        free(s -> k);
//        free(s -> v);
//        free(s -> att);
//        free(s -> logits);
//        free(s -> key_cache);
//        free(s -> value_cache);
    }

    /**
     * initialization: read from checkpoint
     *
     * @param w
     * @param p
     * @param sharedWeights
     */
    private static void checkPointInitWeights(BinFileReader reader, TransformerWeights w, Config p, boolean sharedWeights) {
        w.token_embedding_table = reader.nextFloatArray(p.vocab_size * p.dim);
        w.rms_att_weight = reader.nextFloatArray(p.n_layers * p.dim);
        w.wq = reader.nextFloatArray(p.n_layers * p.dim * p.dim);
        w.wk = reader.nextFloatArray(p.n_layers * p.dim * p.dim);
        w.wv = reader.nextFloatArray(p.n_layers * p.dim * p.dim);
        w.wo = reader.nextFloatArray(p.n_layers * p.dim * p.dim);
        w.rms_ffn_weight = reader.nextFloatArray(p.n_layers * p.dim);
        w.w1 = reader.nextFloatArray(p.n_layers * p.dim * p.hidden_dim);
        w.w2 = reader.nextFloatArray(p.n_layers * p.hidden_dim * p.dim);
        w.w3 = reader.nextFloatArray(p.n_layers * p.dim * p.hidden_dim);
        w.rms_final_weight = reader.nextFloatArray(p.dim);
        int head_size = p.dim / p.n_heads;
        w.freq_cis_real = reader.nextFloatArray(p.seq_len * head_size / 2);
        w.freq_cis_imag = reader.nextFloatArray(p.seq_len * head_size / 2);

        w.wcls = sharedWeights ? w.token_embedding_table : reader.nextFloatArray(p.vocab_size * p.dim);
    }

// ----------------------------------------------------------------------------
// neural net blocks

    private static void accum(float[] a, float[] b, int size) {
        for (int i = 0; i < size; i++) {
            a[i] += b[i];
        }
    }

    private static void rmsnorm(float[] o, float[] x, float[] weight, int weightIndex, int size) {
        // calculate sum of squares
        float ss = 0.0f;
        for (int j = 0; j < size; j++) {
            ss += x[j] * x[j];
        }
        ss /= size;
        ss += 1e-5f;
        ss = 1.0f / (float) Math.sqrt(ss);
        // normalize and scale
        for (int j = 0; j < size; j++) {
            o[j] = weight[weightIndex + j] * (ss * x[j]);
        }
    }

    private static void softmax(float[] x, int index, int size) {
        // find max value (for numerical stability)
        float max_val = x[index]; // index + 0
        for (int i = 1; i < size; i++) {
            if (x[index + i] > max_val) {
                max_val = x[index + i];
            }
        }
        // exp and sum
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            // zzz consider expm1
            x[index + i] = (float) Math.exp(x[index + i] - max_val);
            sum += x[index + i];
        }
        // normalize
        for (int i = 0; i < size; i++) {
            x[index + i] /= sum;
        }
    }

    private static void matmulParallel(float[] xout, float[] x, float[] w, int weightIndex, int n, int d) {
        int sizePerThread = d / THREAD_COUNT;
        CountDownLatch latch = new CountDownLatch(THREAD_COUNT);
        for (int threadId = 0; threadId < THREAD_COUNT; threadId++) {
            // W (d,n) @ x (n,) -> xout (d,)
            final int end = Math.min(d, (threadId + 1) * sizePerThread);
//            LLogger.debug(">>> d " + d + ", n " + n);
            int finalThreadId = threadId;
            new Thread(() -> {
                try {
                    float val;
                    for (int i = finalThreadId * sizePerThread; i < end; i++) {
                        int base = weightIndex + i * n;
                        val = 0.0f;
                        for (int j = 0; j < n; j++) {
                            val += w[base + j] * x[j];
                        }
                        xout[i] = val;
                    }
                } finally {
                    latch.countDown();
                }
            }).start();
        }
        try {
            latch.await();
        } catch (InterruptedException e) {
            LLogger.error("fastMatmul was interrupted");
        }
    }

    private static void matmul(float[] xout, float[] x, float[] w, int weightIndex, int n, int d) {
        matmulParallel(xout, x, w, weightIndex, n, d);
    }

    private static void matmulSimple(float[] xout, float[] x, float[] w, int weightIndex, int n, int d) {
        // W (d,n) @ x (n,) -> xout (d,)
        // by far the most amount of time is spent inside this little function
        int i;
        float val;
        for (i = 0; i < d; i++) {
            val = 0.0f;
            for (int j = 0; j < n; j++) {
                val += w[weightIndex + i * n + j] * x[j];
            }
            xout[i] = val;
        }
    }

    private static void transformer(int token, int pos, Config p, RunState s, TransformerWeights w) {
        // a few convenience variables
        float[] x = s.x;
        int dim = p.dim;
        int hidden_dim = p.hidden_dim;
        int head_size = dim / p.n_heads;

        // copy the token embedding into x
//        float*content_row = &(w.token_embedding_table[token * dim]);
//        memcpy(x, content_row, dim * sizeof( * x));
        System.arraycopy(w.token_embedding_table, token * dim, x, 0, dim);

        // pluck out the "pos" row of freq_cis_real and freq_cis_imag
        int freqCisIndex = pos * head_size / 2;
//        float*freq_cis_real_row = w.freq_cis_real + pos * head_size / 2;
//        float*freq_cis_imag_row = w.freq_cis_imag + pos * head_size / 2;

        // forward all the layers
        for (int l = 0; l < p.n_layers; l++) {

            // attention rmsnorm
            rmsnorm(s.xb, x, w.rms_att_weight, l * dim, dim);

            // qkv matmuls for this position
            matmul(s.q, s.xb, w.wq, l * dim * dim, dim, dim);
            matmul(s.k, s.xb, w.wk, l * dim * dim, dim, dim);
            matmul(s.v, s.xb, w.wv, l * dim * dim, dim, dim);

            // apply RoPE rotation to the q and k vectors for each head
            for (int h = 0; h < p.n_heads; h++) {
                int headIndex = h * head_size;
                // get the q and k vectors for this head
//                float[] q = s.q + h * head_size;
//                float[] k = s.k + h * head_size;
                // rotate q and k by the freq_cis_real and freq_cis_imag
                for (int i = 0; i < head_size; i += 2) {
                    float q0 = s.q[headIndex + i];
                    float q1 = s.q[headIndex + i + 1];
                    float k0 = s.k[headIndex + i];
                    float k1 = s.k[headIndex + i + 1];
                    float fcr = w.freq_cis_real[freqCisIndex + i / 2];
                    float fci = w.freq_cis_imag[freqCisIndex + i / 2];
                    s.q[headIndex + i] = q0 * fcr - q1 * fci;
                    s.q[headIndex + i + 1] = q0 * fci + q1 * fcr;
                    s.k[headIndex + i] = k0 * fcr - k1 * fci;
                    s.k[headIndex + i + 1] = k0 * fci + k1 * fcr;
                }
            }

            // save key,value at this time step (pos) to our kv cache
            int loff = l * p.seq_len * dim; // kv cache layer offset for convenience
//            float*key_cache_row = s.key_cache + loff + pos * dim;
//            memcpy(key_cache_row, s -> k, dim * sizeof( * key_cache_row));

            System.arraycopy(s.k, 0, s.key_cache, loff + pos * dim, dim);

//            float*value_cache_row = s.value_cache + loff + pos * dim;
//            memcpy(value_cache_row, s -> v, dim * sizeof( * value_cache_row));

            System.arraycopy(s.v, 0, s.value_cache, loff + pos * dim, dim);

            // multihead attention. iterate over all heads
            int h;
            for (h = 0; h < p.n_heads; h++) {
                // get the query vector for this head
//                float*q = s -> q + h * head_size;
                int queryIndex = h * head_size;
                // attention scores for this head
                int attentionIndex = h * p.seq_len;
//                float*att = s -> att + h * p.seq_len;
                // iterate over all timesteps, including the current one
                for (int t = 0; t <= pos; t++) {
                    // get the key vector for this head and at this timestep
//                    float*k = s -> key_cache + loff + t * dim + h * head_size;
                    int keyIndex = loff + t * dim + h * head_size;
                    // calculate the attention score as the dot product of q and k
                    float score = 0.0f;
                    for (int i = 0; i < head_size; i++) {
                        score += s.q[queryIndex + i] * s.key_cache[keyIndex + i];
                    }
                    score /= (float) Math.sqrt(head_size);
                    // save the score to the attention buffer
                    s.att[attentionIndex + t] = score;
                }

                // softmax the scores to get attention weights, from 0..pos inclusively
                softmax(s.att, attentionIndex, pos + 1);

                // weighted sum of the values, store back into xb
                int xbIndex = h * head_size;
//                float*xb = s -> xb + h * head_size;
//                memset(xb, 0, head_size * sizeof( float));

                Arrays.fill(s.xb, xbIndex, xbIndex + head_size, 0f);

                for (int t = 0; t <= pos; t++) {
                    // get the value vector for this head and at this timestep
                    int vIndex = loff + t * dim + h * head_size;
//                    float*v = s -> value_cache + loff + t * dim + h * head_size;
                    // get the attention weight for this timestep
                    float a = s.att[attentionIndex + t];
                    // accumulate the weighted value into xb
                    for (int i = 0; i < head_size; i++) {
                        s.xb[xbIndex + i] += a * s.value_cache[vIndex + i];
                    }
                }
            }

            // final matmul to get the output of the attention
            matmul(s.xb2, s.xb, w.wo, l * dim * dim, dim, dim);

            // residual connection back into x
            accum(x, s.xb2, dim);

            // ffn rmsnorm
            rmsnorm(s.xb, x, w.rms_ffn_weight, l * dim, dim);

            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // first calculate self.w1(x) and self.w3(x)
            matmul(s.hb, s.xb, w.w1, l * dim * hidden_dim, dim, hidden_dim);
            matmul(s.hb2, s.xb, w.w3, l * dim * hidden_dim, dim, hidden_dim);

            // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
            for (int i = 0; i < hidden_dim; i++) {
                s.hb[i] = s.hb[i] * (float) (1.0f / (1.0f + Math.exp((-s.hb[i]))));
            }

            // elementwise multiply with w3(x)
            for (int i = 0; i < hidden_dim; i++) {
                s.hb[i] = s.hb[i] * s.hb2[i];
            }

            // final matmul to get the output of the ffn
            matmul(s.xb, s.hb, w.w2, l * dim * hidden_dim, hidden_dim, dim);

            // residual connection
            accum(x, s.xb, dim);
        }

        // final rmsnorm
        rmsnorm(x, x, w.rms_final_weight, 0, dim);

        // classifier into logits
        matmul(s.logits, x, w.wcls, 0, p.dim, p.vocab_size);
    }

// ----------------------------------------------------------------------------
// byte pair encoding (BPE) tokenizer, encodes strings into tokens so we can prompt

    private static int str_lookup(char c1, String[] vocab) {
        // find the first perfect match for str in vocab, return its index or -1 if not found
        for (int i = 0; i < vocab.length; i++) {
            if (vocab[i].length() == 1 && vocab[i].charAt(0) == c1) {
                return i;
            }
        }
        return -1;
    }

    private static int str_lookup(String str, String[] vocab) {
        // find the first perfect match for str in vocab, return its index or -1 if not found
        for (int i = 0; i < vocab.length; i++) {
            if (vocab[i].equals(str)) {
                return i;
            }
        }
        return -1;
    }

    /**
     * @param str
     * @param vocab
     * @param vocab_scores
     * @param max_token_length
     * @param tokens
     * @return number of tokens
     */
    private static int bpe_encode(String str, String[] vocab, float[] vocab_scores, int max_token_length, int[] tokens) {

        // first encode every individual byte in the input string

        int n_tokens = 0; // the number of tokens

        // a temporary buffer to merge two consecutive tokens

        char[] characters = str.toCharArray();
//        char[] str_buffer = new char[max_token_length * 2 + 1]; // *2 for concat, +1 for null terminator
//        System.arraycopy(characters, 0, str_buffer, 0, str.length());

        for (int i = 0; i < str.length(); i++) {
            int id = str_lookup(characters[i], vocab);
            if (id == -1) {
                LLogger.error("Unknown prompt character '" + characters[i] + "'");
                System.exit(1);
            }
            tokens[n_tokens] = id;
            n_tokens++;
        }

        // merge the best consecutive pair each iteration, according the scores in vocab_scores
        while (true) {
            float best_score = -1e10f;
            int best_id = -1;
            int best_idx = -1;

            for (int i = 0; i < n_tokens - 1; i++) {
                // check if we can merge the pair (tokens[i], tokens[i+1])

                String str_buffer = vocab[tokens[i]] + vocab[tokens[i + 1]];
                int id = str_lookup(str_buffer, vocab);
                if (id != -1 && vocab_scores[id] > best_score) {
                    // this merge pair exists in vocab! record its score and position
                    best_score = vocab_scores[id];
                    best_id = id;
                    best_idx = i;
                }
            }

            if (best_idx == -1) {
                break; // we couldn't find any more pairs to merge, so we're done
            }

            // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
            tokens[best_idx] = best_id;
            // delete token at position best_idx+1, shift the entire sequence back 1
            for (int i = best_idx + 1; i < n_tokens - 1; i++) {
                tokens[i] = tokens[i + 1];
            }
            n_tokens--;
        }
        return n_tokens;
    }

// ----------------------------------------------------------------------------
// utilities

    private static long time() {
        return System.currentTimeMillis();
    }

    private static int random_u32() {
        // zzz do we need this?
        return random.nextInt();
    }

    private static float random_f32() { // random float32 in [0,1)
        return random.nextFloat();
    }

    private static int sample(float[] probabilities, int n) {
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

    private static int argmax(float[] v, int n) {
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

    public static void main(String[] args) {
        // poor man's argparse
        String checkpoint;
        float temperature = 0.9f; // e.g. 1.0, or 0.0
        int steps = 256;          // max number of steps to run for, 0: use seq_len
        String prompt = "";      // prompt string

        // 'checkpoint' is necessary arg
        if (args.length < 1) {
            System.out.println("Usage: <COMMAND> <checkpoint_file> [temperature] [steps] [prompt]\n");
            System.exit(1);
        }
        checkpoint = args[0];
        if (args.length >= 2) {
            // optional temperature. 0.0 = (deterministic) argmax sampling. 1.0 = baseline
            temperature = Float.parseFloat(args[1]);
        }
        if (args.length >= 3) {
            steps = Integer.parseInt(args[2]);
        }
        if (args.length >= 4) {
            prompt = args[3];
        }

        // read in the model.bin file

        Config config = new Config();
        TransformerWeights weights = new TransformerWeights();

        long startModelRead = time();

        try (BinFileReader reader = new BinFileReader(MODELS_DIRECTORY + File.separator + checkpoint)) {
            // read in the config header
            config.dim = reader.nextInt(); // transformer dimension
            config.hidden_dim = reader.nextInt(); // for ffn layers
            config.n_layers = reader.nextInt(); // number of layers
            config.n_heads = reader.nextInt(); // number of query heads
            config.n_kv_heads = reader.nextInt(); // number of key/value heads (can be < query heads because of multiquery)
            config.vocab_size = reader.nextInt(); // vocabulary size, usually 256 (byte-level)
            config.seq_len = reader.nextInt(); // max sequence length

            // negative vocab size is hacky way of signaling unshared weights. bit yikes.
            boolean shared_weights = config.vocab_size > 0;
            config.vocab_size = abs(config.vocab_size);

            checkPointInitWeights(reader, weights, config, shared_weights);
        } catch (IOException e) {
            System.exit(1);
        }

        long endModelRead = time();

        LLogger.info("Read model in " + String.format("%.2f", (endModelRead - startModelRead) / 1000d) + " s");

        // right now we cannot run for more than config.seq_len steps
        if (steps <= 0 || steps > config.seq_len) {
            steps = config.seq_len;
        }

        // read in the tokenizer.bin file
        String[] vocab = new String[config.vocab_size];
        float[] vocab_scores = new float[config.vocab_size];

        // read tokenizer

        long startTokenizerRead = time();

        int max_token_length = 0;
        try (BinFileReader reader = new BinFileReader(MODELS_DIRECTORY + File.separator + TOKENIZER_FILE)) {
            max_token_length = reader.nextInt();

            for (int i = 0; i < config.vocab_size; i++) {
                vocab_scores[i] = reader.nextFloat();
                int len = reader.nextInt();
                vocab[i] = reader.nextString(len);
            }
        } catch (IOException e) {
            System.exit(1);
        }

        long endTokenizerRead = time();

        LLogger.info("Read tokenizer in " + String.format("%.2f", (endTokenizerRead - startTokenizerRead) / 1000d) + " s");

        // create and init the application RunState
        RunState state = mallocRunState(config);

        // process the prompt, if any
        int[] prompt_tokens = new int[config.seq_len];
        int num_prompt_tokens = 0;
        if (prompt != null) {
            num_prompt_tokens = bpe_encode(prompt, vocab, vocab_scores, max_token_length, prompt_tokens);
        }

        // start the main loop
        long start = 0;  // used to time our code, only initialized after first iteration
        int next;        // will store the next token in the sequence
        int token = 1;   // init with token 1 (=BOS), as done in Llama-2 sentencepiece tokenizer
        int pos = 0;     // position in the sequence

        Output.emit("<s>\n"); // explicit print the initial BOS token for stylistic symmetry reasons
        while (pos < steps) {

            // forward the transformer to get logits for the next token
            transformer(token, pos, config, state, weights);

            if (pos < num_prompt_tokens) {
                // if we are still processing the input prompt, force the next prompt token
                next = prompt_tokens[pos];
            } else {
                // sample the next token
                if (temperature == 0.0f) {
                    // greedy argmax sampling: take the token with the highest probability
                    next = argmax(state.logits, config.vocab_size);
                } else {
                    // apply the temperature to the logits
                    for (int q = 0; q < config.vocab_size; q++) {
                        state.logits[q] /= temperature;
                    }
                    // apply softmax to the logits to get the probabilities for next token
                    softmax(state.logits, 0, config.vocab_size);
                    // we sample from this distribution to get the next token
                    next = sample(state.logits, config.vocab_size);
                }
            }

            // following BOS token (1), sentencepiece decoder strips any leading whitespace (see PR #89)
            String token_str = (token == 1 && vocab[next].charAt(0) == ' ') ? vocab[next] + 1 : vocab[next];
            Output.emit(token_str);

            // advance forward
            token = next;
            pos++;
            // init our timer here because the first iteration is slow due to memmap
            if (start == 0) {
                start = time();
            }
        }
        Output.emit("\n"); // explicit print the initial BOS token for stylistic symmetry reasons

        // report achieved tok/s
        long end = time();

        LLogger.debug("\nachieved tok/s: " + String.format("%.1f", (steps - 1) / (double) (end - start) * 1000));

        // memory and file handles cleanup
        free_run_state(state);
//        for (int i = 0; i < config.vocab_size; i++) { free(vocab[i]);
//        }
//
//        free(vocab);
//
//        free(vocab_scores);
//        if (prompt_tokens != NULL)
//
//            free(prompt_tokens);
//        if (data != MAP_FAILED)
//
//            munmap(data, file_size);
//        if (fd != -1)
//
//            close(fd);
    }
}
