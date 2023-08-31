package com.lastbot.llama2j;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Handles command line parameters, usage info, error checking, default values
 */
public class CommandLine {
    private static final String CHECKPOINT = "--checkpoint";
    private static final String MODE = "--mode";
    private static final String TEMP = "--temp";
    private static final String TOPP = "--topp";
    private static final String SEED = "--seed";
    private static final String STEPS = "--steps";
    private static final String GPU = "--gpuMem";
    private static final String PROMPT = "--prompt";
    private static final String TOKENIZER = "--tokenizer";

    private final String checkpoint;
    private Mode mode = Mode.CUDA; // CPU, TEST, CUDA
    private float temperature = 0.9f; // e.g. 1.0, or 0.0
    private Float topp = 0.9f; // e.g. 1.0, or 0.0
    private Long seed = null; // e.g. 12345
    private int steps = 256;          // max number of steps to run for, 0: use seq_len
    private String prompt = "One day, Lily met a Shoggoth";      // prompt string
    private String tokenizer = "tokenizer.bin";      // tokenizer file

    private static final long GIGA = 1024L * 1024L * 1024L;

    private long[] gpuMem = null;

    public CommandLine(String[] args) throws IllegalArgumentException {
        Map<String, String> arguments = new HashMap<>();
        for (int i = 0; i < args.length - 1; i += 2) {
            arguments.put(args[i], args[i + 1]);
        }

        try {
            checkpoint = arguments.get(CHECKPOINT);
            if (checkpoint == null) {
                showUsage();
                System.exit(1);
            } else {
                LLogger.info(CHECKPOINT + " " + checkpoint);
            }

            if (arguments.containsKey(MODE)) {
                String s = arguments.get(MODE);
                if (s.equalsIgnoreCase("CPU")) {
                    mode = Mode.CPU;
                } else if (s.equalsIgnoreCase("TEST")) {
                    mode = Mode.TEST;
                } else if (s.equalsIgnoreCase("CUDA")) {
                    mode = Mode.CUDA;
                } else {
                    LLogger.warning("Unknown mode " + MODE + " " + s + ", using default mode");
                }
                LLogger.info(MODE + " " + mode);
            } else {
                LLogger.info(MODE + " " + mode + " (using default)");
            }

            if (arguments.containsKey(TEMP)) {
                temperature = Float.parseFloat(arguments.get(TEMP));
                LLogger.info(TEMP + " " + temperature);
            } else {
                LLogger.info(TEMP + " " + temperature + " (using default)");
            }

            if (arguments.containsKey(TOPP)) {
                topp = Float.parseFloat(arguments.get(TOPP));
                LLogger.info(TOPP + " " + topp);
            } else {
                LLogger.info(TOPP + " " + topp + " (using default)");
            }

            if (arguments.containsKey(SEED)) {
                seed = Long.parseLong(arguments.get(SEED));
                LLogger.info(SEED + " " + seed);
            } else {
                LLogger.info(SEED + " " + seed + " (using default, use current time)");
            }

            if (arguments.containsKey(STEPS)) {
                steps = Integer.parseInt(arguments.get(STEPS));
                LLogger.info(STEPS + " " + steps);
            } else {
                LLogger.info(STEPS + " " + steps + " (using default)");
            }

            if (arguments.containsKey(PROMPT)) {
                prompt = arguments.get(PROMPT);
                LLogger.info(PROMPT + " " + prompt);
            } else {
                LLogger.info(PROMPT + " " + prompt + " (using default)");
            }

            if (arguments.containsKey(TOKENIZER)) {
                tokenizer = arguments.get(TOKENIZER);
                LLogger.info(TOKENIZER + " " + tokenizer);
            } else {
                LLogger.info(TOKENIZER + " " + tokenizer + " (using default)");
            }

            if (arguments.containsKey(GPU)) {
                String gpuString = arguments.get(GPU);
                List<Long> memoryList = new ArrayList<>();
                for (String device : gpuString.split(",")) {
                    int memory = Integer.parseInt(device.trim());
                    memoryList.add(memory * GIGA);
                }
                gpuMem = new long[memoryList.size()];
                StringBuilder s = new StringBuilder();
                for (int i = 0; i < memoryList.size(); i++) {
                    gpuMem[i] = memoryList.get(i);
                    s.append(String.format("%,d", gpuMem[i] / GIGA));
                    if (i < memoryList.size() - 1) {
                        s.append(",");
                    }
                }
                LLogger.info(GPU + " " + s);
            } else {
                LLogger.info(GPU + " null (using default)");
            }

        } catch (Exception e) {
            LLogger.error("Error while parsing command line", e);
            throw new IllegalArgumentException(e);
        }
    }

    public String getCheckpoint() {
        return checkpoint;
    }

    public Mode getMode() {
        return mode;
    }

    public float getTemperature() {
        return temperature;
    }

    public Float getTopp() {
        return topp;
    }

    public Long getSeed() {
        return seed;
    }

    public int getSteps() {
        return steps;
    }

    public String getPrompt() {
        return prompt;
    }

    public String getTokenizer() {
        return tokenizer;
    }

    public long[] getGpuMem() {
        return gpuMem;
    }

    private void showUsage() {
        System.out.println("Usage: <COMMAND> \n " +
                CHECKPOINT + " <checkpoint_file> e.g. llama2_7b.bin\n" +
                TEMP + " <temperature> e.g. 0.9\n" +
                TOPP + " <topp> e.g. 0.9 (top-p in nucleus sampling)\n" +
                SEED + " <seed> e.g. 12345 random seed, default is current time\n" +
                STEPS + " <steps> e.g. 256\n" +
                GPU + " <gpu memory allocation per device> e.g. 17,24,24,24\n" +
                PROMPT + " <prompt> e.g. \"One day, Lily met a Shoggoth\"\n" +
                TOKENIZER + " <tokenizer> e.g. mytokenizer.bin \n");
    }
}
