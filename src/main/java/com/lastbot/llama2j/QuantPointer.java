package com.lastbot.llama2j;

import jcuda.Pointer;

/**
 * Record to manage a CUDA device resident array of quantified values
 */
public record QuantPointer(Quant quant, Pointer pointer, long floatOffset) {
}
