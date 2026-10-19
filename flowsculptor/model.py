import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalDepthwiseBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        # Depthwise separable temporal convolution block.
        # The depthwise conv processes each channel independently over time,
        # and the pointwise conv mixes channel information.
        padding = (kernel_size - 1) * dilation // 2
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, N, C]
        # reshape to [B*N, C, T] so Conv1d operates over the temporal dimension
        b, t, n, c = x.shape
        y = x.permute(0, 2, 3, 1).contiguous().view(b * n, c, t)
        y = self.depthwise(y)
        if y.shape[-1] != t:
            y = y[..., :t]
        y = self.pointwise(y)
        y = self.norm(y)
        y = F.gelu(y)
        y = self.dropout(y)
        y = y.view(b, n, c, t).permute(0, 3, 1, 2).contiguous()
        # residual connection preserves original input and adds temporal features
        return x + y


class TemporalEncoder(nn.Module):
    def __init__(self, channels: int, layers: int = 3, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        # stack dilated temporal blocks to increase receptive field over time
        dilations = [2 ** i for i in range(layers)]
        self.blocks = nn.ModuleList(
            [TemporalDepthwiseBlock(channels, kernel_size, dilation=d, dropout=dropout) for d in dilations]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class SparseAdaptiveGraphConv(nn.Module):
    def __init__(self, num_nodes: int, channels: int, embed_dim: int = 16, topk: int = 20, dropout: float = 0.1):
        super().__init__()
        # adaptive graph convolution uses learnable node embeddings
        # to build a dynamic adjacency matrix for each training step.
        self.num_nodes = num_nodes
        self.topk = max(1, min(int(topk), num_nodes))
        self.node_emb1 = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.1)
        self.node_emb2 = nn.Parameter(torch.randn(embed_dim, num_nodes) * 0.1)
        self.mix_logit = nn.Parameter(torch.tensor(0.0))
        self.proj = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def adaptive_adj(self):
        # compute soft adaptive adjacency from node embeddings
        scores = F.relu(torch.matmul(self.node_emb1, self.node_emb2))
        adj = F.softmax(scores, dim=-1)
        if self.topk < self.num_nodes:
            values, indices = torch.topk(adj, self.topk, dim=-1)
            sparse = torch.zeros_like(adj).scatter_(-1, indices, values)
            adj = sparse / torch.clamp(sparse.sum(dim=-1, keepdim=True), min=1e-12)
        return adj

    def forward(self, x, static_adj=None):
        # x: [B, T, N, C]
        # fuse static adjacency with adaptive adjacency if provided
        adj = self.adaptive_adj()
        if static_adj is not None:
            static_adj = static_adj.to(x.device, x.dtype)
            mix = torch.sigmoid(self.mix_logit)
            adj = mix * static_adj + (1.0 - mix) * adj
            adj = adj / torch.clamp(adj.sum(dim=-1, keepdim=True), min=1e-12)
        y = torch.einsum("ij,btjc->btic", adj, x)
        y = self.proj(y)
        y = self.dropout(F.gelu(y))
        return self.norm(x + y)


class HighFlowGate(nn.Module):
    def __init__(self, num_nodes: int, node_embed_dim: int = 16, hidden_dim: int = 32):
        super().__init__()
        self.node_embedding = nn.Parameter(torch.randn(num_nodes, node_embed_dim) * 0.1)
        self.mlp = nn.Sequential(
            nn.Linear(3 + node_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: [B, T, N, C], normalized traffic.
        traffic = x[..., 0]
        recent_mean = traffic.mean(dim=1)
        recent_max = traffic.max(dim=1).values
        recent_slope = traffic[:, -1] - traffic[:, 0]
        stats = torch.stack([recent_mean, recent_max, recent_slope], dim=-1)
        node = self.node_embedding.unsqueeze(0).expand(x.shape[0], -1, -1)
        # gate predicts high-flow weight per node from recent statistics
        return self.mlp(torch.cat([stats, node], dim=-1))


class HFLiteSTNet(nn.Module):
    """High-flow-aware lightweight spatio-temporal forecasting network.

    Input:  [B, lag, num_nodes, input_dim]
    Output: [B, horizon, num_nodes, output_dim]
    """

    def __init__(
        self,
        num_nodes: int,
        lag: int = 12,
        input_dim: int = 1,
        output_dim: int = 1,
        horizon: int = 12,
        channels: int = 48,
        temporal_layers: int = 3,
        graph_embed_dim: int = 16,
        graph_topk: int = 20,
        dropout: float = 0.1,
        residual_prediction: bool = True,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.lag = lag
        self.horizon = horizon
        self.output_dim = output_dim
        self.residual_prediction = residual_prediction
        # temporal baseline directly models the horizon from the lag window.
        self.temporal_skip = nn.Linear(lag, horizon)
        self.input_proj = nn.Linear(input_dim, channels)
        self.temporal = TemporalEncoder(channels, layers=temporal_layers, dropout=dropout)
        self.graph = SparseAdaptiveGraphConv(num_nodes, channels, graph_embed_dim, graph_topk, dropout)
        self.gate = HighFlowGate(num_nodes, node_embed_dim=graph_embed_dim)
        self.readout = nn.Sequential(
            nn.Linear(channels * 3, channels),
            nn.GELU(),
            nn.LayerNorm(channels),
        )
        self.base_head = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels, horizon * output_dim),
        )
        self.peak_head = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels // 2 if channels >= 2 else channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels // 2 if channels >= 2 else channels, horizon * output_dim),
        )
        self._init_delta_heads()

    def _init_delta_heads(self):
        # Start from the temporal linear baseline; graph/high-flow branches learn only residual corrections.
        for head in (self.base_head, self.peak_head):
            final = head[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def temporal_linear_baseline(self, x):
        # Shared NLinear-style baseline: [B,T,N,D] -> [B,H,N,D].
        traffic = x[..., : self.output_dim].permute(0, 2, 3, 1).contiguous()
        if self.residual_prediction:
            last = traffic[..., -1:]
            baseline = last + self.temporal_skip(traffic - last)
        else:
            baseline = self.temporal_skip(traffic)
        # restore shape to [B, H, N, D]
        return baseline.permute(0, 3, 1, 2).contiguous()

    def _format_head(self, y, batch_size):
        # reshape flattened head output to [B, H, N, D]
        y = y.view(batch_size, self.num_nodes, self.horizon, self.output_dim)
        return y.permute(0, 2, 1, 3).contiguous()

    def forward(self, x, static_adj=None, return_parts: bool = False):
        b = x.shape[0]
        h = self.input_proj(x)
        h = self.temporal(h)
        h = self.graph(h, static_adj=static_adj)
        h_last = self.readout(torch.cat([h[:, -1], h.mean(dim=1), h.max(dim=1).values], dim=-1))

        base = self._format_head(self.base_head(h_last), b)
        peak = self._format_head(self.peak_head(h_last), b)
        gate = self.gate(x).unsqueeze(1)
        delta = base + gate * peak
        pred = self.temporal_linear_baseline(x) + delta

        if return_parts:
            return pred, {"base": base, "peak": peak, "gate": gate, "delta": delta, "skip": self.temporal_linear_baseline(x)}
        return pred


class HFSTFormer(nn.Module):
    """Factorized spatio-temporal Transformer for traffic forecasting.

    The temporal encoder models each node's history independently, while the
    spatial encoder lets all sensor states interact before horizon-specific
    decoding. This is intentionally stronger than the lightweight graph model.
    """

    def __init__(
        self,
        num_nodes: int,
        lag: int = 12,
        input_dim: int = 1,
        output_dim: int = 1,
        horizon: int = 12,
        channels: int = 96,
        temporal_layers: int = 2,
        spatial_layers: int = 2,
        attention_heads: int = 4,
        ffn_ratio: float = 2.0,
        dropout: float = 0.1,
        residual_prediction: bool = True,
        high_flow_branch: bool = False,
        high_flow_gate_hidden: int = 64,
        high_flow_gate_bias: float = -2.0,
        use_linear_baseline: bool = True,
        use_temporal_encoder: bool = True,
        use_spatial_encoder: bool = True,
        use_readout_fusion: bool = True,
        use_horizon_decoder: bool = True,
    ):
        super().__init__()
        if channels % attention_heads != 0:
            raise ValueError("channels must be divisible by attention_heads.")
        self.num_nodes = num_nodes
        self.lag = lag
        self.horizon = horizon
        self.output_dim = output_dim
        self.residual_prediction = residual_prediction
        self.high_flow_branch = bool(high_flow_branch)
        self.use_linear_baseline = bool(use_linear_baseline)
        self.use_temporal_encoder = bool(use_temporal_encoder)
        self.use_spatial_encoder = bool(use_spatial_encoder)
        self.use_readout_fusion = bool(use_readout_fusion)
        self.use_horizon_decoder = bool(use_horizon_decoder)

        if self.use_linear_baseline:
            # temporal linear baseline from lag to horizon
            self.temporal_skip = nn.Linear(lag, horizon)
        self.input_proj = nn.Linear(input_dim, channels)
        # node and lag embeddings inject node-specific and timestep-specific biases
        self.node_embedding = nn.Parameter(torch.randn(num_nodes, channels) * 0.02)
        self.lag_embedding = nn.Parameter(torch.randn(lag, channels) * 0.02)
        if self.use_horizon_decoder:
            self.horizon_embedding = nn.Parameter(torch.randn(horizon, channels) * 0.02)

        dim_feedforward = int(channels * ffn_ratio)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=attention_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        spatial_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=attention_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        if self.use_temporal_encoder:
            self.temporal_encoder = nn.TransformerEncoder(
                temporal_layer,
                num_layers=temporal_layers,
                enable_nested_tensor=False,
            )
        else:
            self.temporal_encoder = nn.Identity()
        if self.use_spatial_encoder:
            self.spatial_encoder = nn.TransformerEncoder(
                spatial_layer,
                num_layers=spatial_layers,
                enable_nested_tensor=False,
            )
        else:
            self.spatial_encoder = nn.Identity()

        readout_input_dim = channels * 3 if self.use_readout_fusion else channels
        self.temporal_readout = nn.Sequential(
            nn.Linear(readout_input_dim, channels),
            nn.GELU(),
            nn.LayerNorm(channels),
        )
        if self.use_horizon_decoder:
            self.horizon_decoder = nn.Sequential(
                nn.LayerNorm(channels),
                nn.Linear(channels, channels),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(channels, output_dim),
            )
        else:
            self.direct_horizon_head = nn.Sequential(
                nn.LayerNorm(channels),
                nn.Linear(channels, channels),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(channels, horizon * output_dim),
            )
        if self.high_flow_branch:
            # high-flow branch learns additional correction gated by recent traffic stats
            self.high_flow_gate = nn.Sequential(
                nn.Linear(channels + 3, high_flow_gate_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(high_flow_gate_hidden, output_dim),
                nn.Sigmoid(),
            )
            self.high_flow_head = nn.Sequential(
                nn.LayerNorm(channels),
                nn.Linear(channels, max(channels // 2, output_dim)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(max(channels // 2, output_dim), output_dim),
            )
            nn.init.constant_(self.high_flow_gate[-2].bias, high_flow_gate_bias)
        self._init_delta_head()

    def _init_delta_head(self):
        final = self.horizon_decoder[-1] if self.use_horizon_decoder else self.direct_horizon_head[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        if self.high_flow_branch:
            high_final = self.high_flow_head[-1]
            nn.init.zeros_(high_final.weight)
            nn.init.zeros_(high_final.bias)

    def temporal_linear_baseline(self, x):
        # compute baseline prediction from past traffic without learned horizon decoder.
        if not self.use_linear_baseline:
            return x.new_zeros(x.shape[0], self.horizon, self.num_nodes, self.output_dim)
        traffic = x[..., : self.output_dim].permute(0, 2, 3, 1).contiguous()
        if self.residual_prediction:
            last = traffic[..., -1:]
            baseline = last + self.temporal_skip(traffic - last)
        else:
            baseline = self.temporal_skip(traffic)
        return baseline.permute(0, 3, 1, 2).contiguous()

    def forward(self, x, static_adj=None, return_parts: bool = False):
        del static_adj
        b, t, n, _ = x.shape
        # input projection and position/node embeddings
        h = self.input_proj(x)
        h = h + self.lag_embedding.view(1, t, 1, -1)
        h = h + self.node_embedding.view(1, 1, n, -1)

        # temporal encoder processes each node independently over time
        h = h.permute(0, 2, 1, 3).contiguous().view(b * n, t, -1)
        h = self.temporal_encoder(h)
        h = h.view(b, n, t, -1).permute(0, 2, 1, 3).contiguous()

        # readout optionally fuses last, mean, and max node states
        if self.use_readout_fusion:
            readout = torch.cat([h[:, -1], h.mean(dim=1), h.max(dim=1).values], dim=-1)
        else:
            readout = h[:, -1]
        node_state = self.temporal_readout(readout)
        node_state = node_state + self.node_embedding.view(1, n, -1)

        # spatial encoder mixes information across nodes
        node_state = self.spatial_encoder(node_state)

        # horizon decoder predicts per-horizon correction delta
        if self.use_horizon_decoder:
            horizon_state = node_state.unsqueeze(1) + self.horizon_embedding.view(1, self.horizon, 1, -1)
            delta = self.horizon_decoder(horizon_state)
        else:
            horizon_state = node_state.unsqueeze(1).expand(-1, self.horizon, -1, -1).contiguous()
            delta = self.direct_horizon_head(node_state)
            delta = delta.view(b, n, self.horizon, self.output_dim).permute(0, 2, 1, 3).contiguous()
        skip = self.temporal_linear_baseline(x)
        pred = skip + delta
        high_delta = None
        high_gate = None

        if self.high_flow_branch:
            # high-flow branch adds a specialized residual correction
            traffic = x[..., 0]
            recent_mean = traffic.mean(dim=1)
            recent_max = traffic.max(dim=1).values
            recent_slope = traffic[:, -1] - traffic[:, 0]
            high_stats = torch.stack([recent_mean, recent_max, recent_slope], dim=-1)
            high_stats = high_stats.unsqueeze(1).expand(-1, self.horizon, -1, -1)
            high_gate = self.high_flow_gate(torch.cat([horizon_state, high_stats], dim=-1))
            high_delta = self.high_flow_head(horizon_state)
            pred = pred + high_gate * high_delta

        if return_parts:
            parts = {"delta": delta, "skip": skip, "node_state": node_state}
            if self.high_flow_branch:
                parts.update({"high_gate": high_gate, "high_delta": high_delta})
            return pred, parts
        return pred
