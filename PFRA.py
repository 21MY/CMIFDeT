import torch
import torch.nn as nn
import torch.nn.functional as F

class PFRA(nn.Module):


    def __init__(self, channels, reduction=8):
        super(PFRA, self).__init__()
        C = channels
        C_s = C // reduction  # reduced dimension for spatial attention

        # ---- Stage 2A: IR -> RGB (Spatial Attention) ----
        self.q_ir = nn.Conv2d(C, C_s, kernel_size=1)   # query from IR
        self.k_rgb = nn.Conv2d(C, C_s, kernel_size=1)  # key from RGB
        self.v_rgb = nn.Conv2d(C, C, kernel_size=1)    # value from RGB
        self.proj_space = nn.Conv2d(C, C, kernel_size=1)

        # ---- Stage 2B: RGB -> IR (Channel Attention) ----
        self.q_rgb = nn.Conv2d(C, C, kernel_size=1)    # query from RGB (renamed from q_rgb_down)
        self.k_ir = nn.Conv2d(C, C, kernel_size=1)     # key from IR
        self.v_ir = nn.Conv2d(C, C, kernel_size=1)     # value from IR
        self.proj_channel = nn.Conv2d(C, C, kernel_size=1)

        # ---- Stage 3: Progressive Gated Refinement (shared FFN) ----
        self.gated_ffn = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(C, C, kernel_size=3, padding=1),
        )
        self.ffn_gate = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.proj_ffn = nn.Conv2d(C, C, kernel_size=1)

    def forward(self, F_rgb, F_ir):

        B, C, H, W = F_rgb.shape
        N = H * W

        # ================ Stage 2A: IR -> RGB (Spatial Attention) ================
        Q_s = self.q_ir(F_ir)          # (B, C_s, H, W)
        K_s = self.k_rgb(F_rgb)        # (B, C_s, H, W)
        V_s = self.v_rgb(F_rgb)        # (B, C, H, W)

        # Flatten spatial dimensions
        Q_s_flat = Q_s.view(B, -1, N)  # (B, C_s, N)
        K_s_flat = K_s.view(B, -1, N)  # (B, C_s, N)
        V_s_flat = V_s.view(B, -1, N)  # (B, C, N)

        # Spatial affinity: A_s = softmax(Q^T K / sqrt(C_s))
        A_s = torch.bmm(Q_s_flat.transpose(1, 2), K_s_flat) / (C_s ** 0.5)  # (B, N, N)
        A_s = F.softmax(A_s, dim=-1)

        # Apply attention to V: (B, C, N) = V * A_s^T
        F_rgb_attended = torch.bmm(V_s_flat, A_s.transpose(1, 2))  # (B, C, N)
        F_rgb_attended = F_rgb_attended.view(B, C, H, W)

        # Residual connection
        F_rgb_mid = F_rgb + self.proj_space(F_rgb_attended)

        # ================ Stage 2B: RGB -> IR (Channel Attention) ================
        Q_c = self.q_rgb(F_rgb)        # (B, C, H, W)
        K_c = self.k_ir(F_ir)          # (B, C, H, W)
        V_c = self.v_ir(F_ir)          # (B, C, H, W)

        # Flatten spatial and perform channel-wise attention
        Q_c_flat = Q_c.view(B, C, N)   # (B, C, N)
        K_c_flat = K_c.view(B, C, N)   # (B, C, N)
        V_c_flat = V_c.view(B, C, N)   # (B, C, N)

        # Channel affinity: A_c = softmax(Q * K^T / sqrt(C))
        A_c = torch.bmm(Q_c_flat, K_c_flat.transpose(1, 2)) / (C ** 0.5)  # (B, C, C)
        A_c = F.softmax(A_c, dim=-1)

        # Apply to V: (B, C, N) = A_c * V
        F_ir_attended = torch.bmm(A_c, V_c_flat)  # (B, C, N)
        F_ir_attended = F_ir_attended.view(B, C, H, W)

        # Residual connection
        F_ir_mid = F_ir + self.proj_channel(F_ir_attended)

        # ================ Stage 3: Progressive Gated Refinement (shared FFN) ================
        # RGB branch
        ffn_out_rgb = self.gated_ffn(F_rgb_mid)
        gate_rgb = self.ffn_gate(F_rgb_mid)
        F_rgb_refined = F_rgb_mid + self.proj_ffn(ffn_out_rgb * gate_rgb)

        # IR branch
        ffn_out_ir = self.gated_ffn(F_ir_mid)
        gate_ir = self.ffn_gate(F_ir_mid)
        F_ir_refined = F_ir_mid + self.proj_ffn(ffn_out_ir * gate_ir)

        return F_rgb_refined, F_ir_refined