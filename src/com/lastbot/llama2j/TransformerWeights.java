package com.lastbot.llama2j;

public class TransformerWeights {  // token embedding table
    float[] token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float[] rms_att_weight; // (layer, dim) rmsnorm weights
    float[] rms_ffn_weight; // (layer, dim)
    // weights for matmuls
    float[] wq; // (layer, dim, dim)
    float[] wk; // (layer, dim, dim)
    float[] wv; // (layer, dim, dim)
    float[] wo; // (layer, dim, dim)
    // weights for ffn
    float[] w1; // (layer, hidden_dim, dim)
    float[] w2; // (layer, dim, hidden_dim)
    float[] w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float[] rms_final_weight; // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    float[] freq_cis_real; // (seq_len, dim/2)
    float[] freq_cis_imag; // (seq_len, dim/2)
    // (optional) classifier weights for the logits, on the last layer
    float[] wcls;
}
