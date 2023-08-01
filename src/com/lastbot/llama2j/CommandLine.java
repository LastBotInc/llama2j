package com.lastbot.llama2j;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class CommandLine {
    private static final String CHECKPOINT = "--checkpoint";
    private static final String TEMP = "--temp";
    private static final String STEPS = "--steps";
    private static final String GPU = "--gpuMem";
    private static final String PROMPT = "--prompt";

    private String checkpoint;
    private float temperature = 0.9f; // e.g. 1.0, or 0.0
    private int steps = 256;          // max number of steps to run for, 0: use seq_len
    private String prompt = "One day, Lily met a Shoggoth";      // prompt string

    private int[] gpuMem = null;

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

            if (arguments.containsKey(TEMP)) {
                temperature = Float.parseFloat(arguments.get(TEMP));
                LLogger.info(TEMP + " " + temperature);
            } else {
                LLogger.info(TEMP + " " + temperature + " (using default)");
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

            if (arguments.containsKey(GPU)) {
                String gpuString = arguments.get(GPU);
                List<Integer> memoryList = new ArrayList<>();
                for (String device : gpuString.split(",")) {
                    int memory = Integer.parseInt(device.trim());
                    memoryList.add(memory);
                }
                gpuMem = new int[memoryList.size()];
                String s = "";
                for (int i = 0; i < memoryList.size(); i++) {
                    gpuMem[i] = memoryList.get(i);
                    s += gpuMem[i] + (i < memoryList.size() - 1 ? "," : "");
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

    public float getTemperature() {
        return temperature;
    }

    public int getSteps() {
        return steps;
    }

    public String getPrompt() {
        return prompt;
    }

    public int[] getGpuMem() {
        return gpuMem;
    }

    private void showUsage() {
        System.out.println("Usage: <COMMAND> \n " +
                CHECKPOINT + " <checkpoint_file> e.g. llama2_7b.bin\n" +
                TEMP + " <temperature> e.g. 0.9\n" +
                STEPS + " <steps> e.g. 256\n" +
                GPU + " <gpu memory allocation per device> e.g. 17,24,24,24\n" +
                PROMPT + " <prompt> e.g. \"One day, Lily met a Shoggoth\"\n");
    }
}
