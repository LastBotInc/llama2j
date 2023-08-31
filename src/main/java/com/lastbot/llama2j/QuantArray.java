package com.lastbot.llama2j;

/**
 * Record to manage a CPU resident array of quantified values
 */
public record QuantArray(Quant quant, byte[] data) {
}
