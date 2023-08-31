package com.lastbot.llama2j;

/**
 * This class sets global limits.
 */
public class Limits {
    /**
     * Largest array size. Please note that arrays of size Integer.MAX_VALUE do not work.
     */
    public static final int ARRAY_MAX_SIZE = 32 * (Integer.MAX_VALUE / 32);
}
