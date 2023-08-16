package com.lastbot.llama2j;

import java.io.Closeable;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class Tokenizer implements Closeable {
    private static final Charset CHARSET = StandardCharsets.UTF_8;
    private static final boolean DEBUG = false;

    // read in the tokenizer.bin file
    private final String[] vocab;
    private final float[] vocab_scores;

    private final TokenIndex[] sorted_vocab;
    private final int vocabSize;

    private record TokenIndex(String str, int id) implements Comparable<TokenIndex> {
        @Override
        public int compareTo(TokenIndex o) {
            return this.str.compareTo(o.str);
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            TokenIndex that = (TokenIndex) o;
            return Objects.equals(str, that.str);
        }

        @Override
        public int hashCode() {
            return Objects.hash(str);
        }
    }

    private static int str_lookup(String str, TokenIndex[] sorted_vocab) {
        // efficiently find the perfect match for str in vocab, return its index or -1 if not found
        TokenIndex tok = new TokenIndex(str, 0); // acts as the key to search for
        int index = Arrays.binarySearch(sorted_vocab, tok);
        if (index < 0) {
            return -1;
        }
        int id = sorted_vocab[index].id;
        return id;
    }

    /**
     * byte pair encoding (BPE) tokenizer, encodes strings into tokens so we can prompt
     */
    public Tokenizer(String tokenizerFilePath, int vocabSize) {
        // read in the tokenizer.bin file
        this.vocabSize = vocabSize;
        this.vocab = new String[vocabSize];
        this.vocab_scores = new float[vocabSize];
        this.sorted_vocab = new TokenIndex[vocabSize];

        long startTokenizerRead = System.currentTimeMillis();

        try (BinFileReader reader = new BinFileReader(tokenizerFilePath)) {
            int max_token_length = reader.nextInt();

            for (int i = 0; i < vocabSize; i++) {
                vocab_scores[i] = reader.nextFloat();
                int len = reader.nextInt();
                vocab[i] = reader.nextString(len, CHARSET);
                sorted_vocab[i] = new TokenIndex(vocab[i], i);

                if (DEBUG) {
                    LLogger.debug("[" + i + "] " + String.format("%,.5f", vocab_scores[i]) + " |" + vocab[i] + "|");
                }
            }
        } catch (
                IOException e) {
            System.exit(1);
        }
        Arrays.sort(sorted_vocab);

        long endTokenizerRead = System.currentTimeMillis();

        LLogger.info("Read tokenizer in " + String.format("%.2f", (endTokenizerRead - startTokenizerRead) / 1000d) + " s");
    }

    private static int str_lookup(char c1, String[] vocab) {
        // find the first perfect match for str in vocab, return its index or -1 if not found
        for (int i = 0; i < vocab.length; i++) {
            if (vocab[i].length() == 1 && vocab[i].charAt(0) == c1) {
                return i;
            }
        }
        return -1;
    }

//    private static int str_lookup(String str, String[] vocab) {
//        // find the first perfect match for str in vocab, return its index or -1 if not found
//        for (int i = 0; i < vocab.length; i++) {
//            if (vocab[i].equals(str)) {
//                return i;
//            }
//        }
//        return -1;
//    }

    public int[] bpe_encode(String prompt) {
        int[] tokens = new int[prompt.length() * 2 + 1];
        // first encode every individual byte in the input string

        // add_dummy_prefix is true by default
        tokens[0] = str_lookup(" ", sorted_vocab);
        int n_tokens = 1; // the number of tokens

        char[] characters = prompt.toCharArray();

        for (int i = 0; i < prompt.length(); i++) {
//            int id2 = str_lookup(Character.toString(characters[i]), sorted_vocab);
            int id = str_lookup(characters[i], vocab);
//            if (id != id2) {
//                LLogger.error("toked id " + id + " id2 " + id2);
//            }

            if (id != -1) {
                // we found this codepoint in vocab, add it as a token
                tokens[n_tokens++] = id;
            } else {
                // byte_fallback encoding: just encode each byte as a token
                // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
                // so the individual bytes only start at index 3

                byte[] bytes = Character.toString(characters[i]).getBytes(CHARSET);
                for (byte b : bytes) {
                    tokens[n_tokens++] = b + 3;
                }
            }
        }

        // merge the best consecutive pair each iteration, according the scores in vocab_scores
        while (true) {
            float best_score = -1e10f;
            int best_id = -1;
            int best_idx = -1;

            for (int i = 0; i < n_tokens - 1; i++) {
                // check if we can merge the pair (tokens[i], tokens[i+1])

                String str_buffer = vocab[tokens[i]] + vocab[tokens[i + 1]];
                int id = str_lookup(str_buffer, sorted_vocab);
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
        int[] result = new int[n_tokens];
        System.arraycopy(tokens, 0, result, 0, n_tokens);
        return result;
    }

    public static void main(String[] args) {
        String token_str = "<0x40>";
        Matcher matcher = RAW_BYTE_TOKEN_PATTERN.matcher(token_str);
        String output;
        if (matcher.matches()) {
            // ok this token is a raw byte token, carefuly to only print printable chars or whitespace
            // some of the other bytes can be various control codes, backspace, etc. => skip
            String hexString = matcher.group(1);
            try {
                int value = Integer.parseInt(hexString, 16);
                char c = (char) (value & 0xFF);
                output = Character.isISOControl(c) ? null : Character.toString(c);
            } catch (Exception e) {
                LLogger.error("unexpected", e);
                output = null;
            }
        }
        else {
            output = token_str;
        }
        LLogger.debug("output |" + output + "|");
    }

    private static final Pattern RAW_BYTE_TOKEN_PATTERN = Pattern.compile("<0x([\\da-fA-F]*)>");

    public String bpe_decode(int token, int next) {
        // following BOS token (1), sentencepiece decoder strips any leading whitespace (see PR #89)
        String token_str = (token == 1 && vocab[next].charAt(0) == ' ') ? vocab[next].substring(1) : vocab[next];

        // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
        Matcher matcher = RAW_BYTE_TOKEN_PATTERN.matcher(token_str);
        if (matcher.matches()) {
            // ok this token is a raw byte token, carefully to only print printable chars or whitespace
            // some of the other bytes can be various control codes, backspace, etc. => skip
            String hexString = matcher.group(1);
            try {
                int value = Integer.parseInt(hexString, 16);
                char c = (char) (value & 0xFF);
                return Character.isISOControl(c) ? null : Character.toString(c);
            } catch (Exception e) {
                return null;
            }
        }
        return token_str;
    }

    @Override
    public void close() {

    }
}
