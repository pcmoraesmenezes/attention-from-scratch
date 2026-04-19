import torch
import torch.nn as nn
from attention import MultiHeadAttention

def test_parity():
    embed_dim = 512
    num_heads = 8
    seq_len = 10
    batch_size = 2


    my_mha = MultiHeadAttention(embed_dim, num_heads, dropout=0.0).eval()
    pt_mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, batch_first=True).eval()
    with torch.no_grad():
        in_proj_weight = torch.cat([
            my_mha.w_query.weight,
            my_mha.w_key.weight,
            my_mha.w_value.weight
        ], dim=0)
        
        in_proj_bias = torch.cat([
            my_mha.w_query.bias,
            my_mha.w_key.bias,
            my_mha.w_value.bias
        ], dim=0)

        pt_mha.in_proj_weight.copy_(in_proj_weight)
        pt_mha.in_proj_bias.copy_(in_proj_bias)
        
        pt_mha.out_proj.weight.copy_(my_mha.w_out.weight)
        pt_mha.out_proj.bias.copy_(my_mha.w_out.bias)

    x = torch.randn(batch_size, seq_len, embed_dim)

    with torch.no_grad():
        my_out, my_attn = my_mha(x)
        pt_out, pt_attn = pt_mha(x, x, x, average_attn_weights=False)

    diff_out = torch.max(torch.abs(my_out - pt_out)).item()
    diff_attn = torch.max(torch.abs(my_attn - pt_attn)).item()

    print(f"--- Parity Check Results ---")
    print(f"Max Output Difference: {diff_out:.2e}")
    print(f"Max Attn Difference:   {diff_attn:.2e}")

    assert diff_out < 1e-6, f"Output difference too high: {diff_out}"
    print("✅ Parity Check Passed!")

if __name__ == "__main__":
    test_parity()
