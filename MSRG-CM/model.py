class CrossWindowSparseAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4,max_length=6000, window_size=4, num_neighbors=8):
        super().__init__()
        self.window_size = window_size
        self.num_neighbors = num_neighbors
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.num_heads = num_heads
        # 投影层
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.eval_save_mode = False  # 评估保存模式开关
        self.attention_archive = []  # 存储注意力数据
        self.max_save_samples = 50   # 最大保存样本数

    def _window_partition(self, x):
        """将序列分割为窗口"""
        B, num_heads, T, H = x.shape  # B: batch size, num_heads: number of heads, T: sequence length, H: hidden dimension
        pad_len = (self.window_size - T % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_len))  # padding sequence length to be a multiple of window size
        
        # Adjust the shape to fit the window partition
        return x.reshape(B, num_heads, -1, self.window_size, H)  # (B, num_heads, N, W, H)

    def _get_neighbors(self, x):
        """获取每个窗口的邻域窗口"""
        B, num_heads, N, W, H = x.shape  # N: number of windows, W: window size, H: hidden dimension
        K = self.num_neighbors
        
        # 镜像填充窗口维度
        x_padded = F.pad(x, (0, 0, 0, 0, K, K), mode='reflect')  # padding both sides of window dimension
        
        # 生成滑动窗口索引
        indices = torch.arange(N, device=x.device).view(N, 1) + torch.arange(-K, K + 1, device=x.device)
        indices = torch.clamp(indices, 0, N + 2 * K - 1)
        
        # Now we ensure the padding and indexing works for each head
        return x_padded[:, :, indices]  # (B, num_heads, N, 2K+1, W, H)

    def forward(self, query, key, value):
        B, T, H = query.shape
        
        # 投影变换
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        # 将每个投影后的张量分成多个头
        Q = Q.view(B, T, self.num_heads, H // self.num_heads).transpose(1, 2)  # (B, num_heads, T, H // num_heads)
        K = K.view(B, T, self.num_heads, H // self.num_heads).transpose(1, 2)  # (B, num_heads, T, H // num_heads)
        V = V.view(B, T, self.num_heads, H // self.num_heads).transpose(1, 2)  # (B, num_heads, T, H // num_heads)

       # 窗口划分
        Q_windows = self._window_partition(Q)  # (B, num_heads, N, W, H)
        K_neighbors = self._get_neighbors(self._window_partition(K))  # (B, num_heads, N, 2K+1, W, H)
        V_neighbors = self._get_neighbors(self._window_partition(V))

        # 调整维度顺序
        Q_exp = Q_windows.unsqueeze(4)  # (B, num_heads, N, W, 1, H // num_heads)
        K_neighbors = K_neighbors.permute(0, 1, 2, 4, 3, 5)  # (B, num_heads, N, 2K+1, W, H // num_heads)

        # 计算注意力分数
        attn = torch.matmul(Q_exp, K_neighbors.transpose(-1, -2))  # (B, num_heads, N, W, 1, 2K+1)
        attn = attn / (H ** 0.5)
        attn = F.softmax(attn, dim=-1)

        # 聚合Value
        V_neighbors = V_neighbors.permute(0, 1, 2, 4, 3, 5)  # (B, num_heads, N, 2K+1, W, H // num_heads)
        output = torch.matmul(attn, V_neighbors)  # (B, num_heads, N, W, 1, H // num_heads)
        output = output.squeeze(4).reshape(B, -1, H)[:, :T, :]  # (B, T, H)

  

        # 评估模式保存逻辑
        if self.eval_save_mode and len(self.attention_archive) < self.max_save_samples:
            sample = {
                'attention': attn.detach().cpu().numpy(),
                'query': query.detach().cpu().numpy(),
                'key': key.detach().cpu().numpy(),
                'timestamp': time.time()
            }
            self.attention_archive.append(sample)

        return self.out_proj(output), attn

    


class DenseGINLayer(nn.Module):
    def __init__(self, hidden_dim, eps=0.0):
        super().__init__()
        self.eps = nn.Parameter(torch.tensor(eps, dtype=torch.float))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, X, A):
        X_hc = X.transpose(1, 2)            # (g,H,C)
        agg_hc = torch.matmul(X_hc, A.t())  # (g,H,C)
        agg = agg_hc.transpose(1, 2)        # (g,C,H)
        return self.mlp((1.0 + self.eps) * X + agg)


class ChannelAttnPool(nn.Module):
    """X: (g,C,H) -> (g,H)"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, X):
        w = torch.softmax(self.score(X).squeeze(-1), dim=1)   # (g,C)
        return (w.unsqueeze(-1) * X).sum(dim=1)               # (g,H)


class MultiScaleGCN(nn.Module):
    def __init__(
        self,
        num_channels,
        hidden_dim,
        num_layers,
        num_scales,
        num_heads=4,
        q_low=0.3,
        q_high=0.8,
       
        use_ema_graph=False,
        ema_beta=0.9,
        use_attn_pool=True,
    ):
        super().__init__()
        
        self.C = num_channels
        self.H = hidden_dim
        self.num_layers = num_layers
        self.num_scales = num_scales
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_low = q_low
        self.q_high = q_high
        self.use_ema_graph = use_ema_graph
        self.ema_beta = ema_beta

        # =========== 核心修改: 内置 CNN 特征提取 ===========
        # 将 3000 个时间点压缩为约 187 个高维特征点
        self.patch_embedding = nn.Sequential(
            nn.Conv1d(1, hidden_dim // 2, kernel_size=32, stride=4, padding=16),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(0.3), # <--- 关键！防止浅层特征过拟合
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3)  # <--- 关键！
        )
        # ===============================================

        if self.use_ema_graph:
            self.register_buffer("A_ema", torch.zeros(num_channels, num_channels))
            self.register_buffer("A_ema_inited", torch.tensor(0, dtype=torch.uint8))

        self.graph_learner = self.DynamicGraphLearner(hidden_dim, num_heads)

        # 加入 LayerNorm 解决梯度 NaN 问题
        self.convs = nn.ModuleList([DenseGINLayer(hidden_dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)]) 
        self.skip_cons = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)])

        self.use_attn_pool = use_attn_pool
        self.pool = ChannelAttnPool(hidden_dim) if use_attn_pool else None

    class DynamicGraphLearner(nn.Module):
        def __init__(self, hidden_dim, num_heads):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = hidden_dim // num_heads
            
            # [修改 1] bias=False
            # 计算角度相似度时，去掉偏置项会让向量的原点对齐，计算更纯粹
            self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.key   = nn.Linear(hidden_dim, hidden_dim, bias=False)

        def forward(self, x):
            # x shape: (Batch*Time, C, H) -> 例如 (48000, 10, 128)
            
            # 1. 聚合: 必须保留这行，防止报错！
            x_mean = x.mean(dim=0) # (C, H) -> (10, 128)
            
            # [建议] 加上 LayerNorm，让特征更稳定
            x_mean = F.layer_norm(x_mean, x_mean.shape[-1:])

            # 2. 投影
            q = self.query(x_mean).view(x_mean.shape[0], self.num_heads, self.head_dim)
            k = self.key(x_mean).view(x_mean.shape[0], self.num_heads, self.head_dim)
            
            # 3. 维度调整: (Heads, C, Head_Dim)
            q = q.permute(1, 0, 2)
            k = k.permute(1, 0, 2)
            
            # [修改 2] 核心：L2 归一化 (L2 Normalization)
            # 这一步把向量长度都变成 1，之后的矩阵乘法就等于计算 Cosine 相似度
            # 这样无论受试者脑电幅度大还是小，都不会影响构图
            q = F.normalize(q, p=2, dim=-1)
            k = F.normalize(k, p=2, dim=-1)
            
            # [修改 3] 温度系数 (Temperature)
            # Cosine 的范围是 [-1, 1]，如果不除以小数值，softmax 出来的概率会太平滑（所有点都连在一起）
            # 0.1 是一个经验值，让图结构更加锐利
            temperature = 0.1
            logits = torch.matmul(q, k.transpose(-2, -1)) / temperature
            
            attn = F.softmax(logits, dim=-1)
            
            # [建议] 加一个 Dropout 防止对某些连接过拟合
            # 既然是解决方差大，Dropout 很有用
            attn = F.dropout(attn, p=0.2, training=self.training)
            
            return attn.mean(dim=0)

    def build_multi_scale_adj(self, X_graph):
        A = self.graph_learner(X_graph)          
        A = A + A.transpose(0, 1) 
        A = torch.nan_to_num(A, nan=0.0)
        
        if self.use_ema_graph:
            if int(self.A_ema_inited.item()) == 0:
                self.A_ema.copy_(A.detach())
                self.A_ema_inited.fill_(1)
            else:
                self.A_ema.mul_(self.ema_beta).add_((1.0 - self.ema_beta) * A.detach())
            A_used = self.A_ema
        else:
            A_used = A

        device = X_graph.device
        qs = torch.linspace(self.q_low, self.q_high, self.num_scales, device=device)
        thrs = torch.quantile(A_used.flatten(), qs)

        mats = []
        for t in thrs:
            mask = (A_used > t).float()
            # 归一化处理，防止数值爆炸
            deg = mask.sum(1)
            deg_inv_sqrt = torch.pow(deg + 1e-6, -0.5)
            deg_mat = torch.diag(deg_inv_sqrt)
            norm_adj = torch.mm(torch.mm(deg_mat, mask * A_used), deg_mat)
            mats.append(norm_adj)
        return mats

    def forward(self, x):
        # x: (B, C, T)
        B, C, T_raw = x.shape
        x = torch.nan_to_num(x, nan=0.0)

        # 1. 变形: (B*C, 1, T)
        x_flat = x.view(B * C, 1, T_raw)
        
        
        # 输出: (B*C, H, T_new)  T_new ≈ 187
        x_emb = self.patch_embedding(x_flat)
        
        _, H, T_new = x_emb.shape
        
        # 3. 变形回 GiN 格式: (B, T_new, C, H)
        x_emb = x_emb.view(B, C, H, T_new).permute(0, 3, 1, 2)
        X_all = x_emb.reshape(B * T_new, C, H)
        
        adj_scales = self.build_multi_scale_adj(X_all)

        X = X_all 
        residual = X
        outs = []

        for layer_idx in range(self.num_layers):
            scale = min(layer_idx, self.num_scales - 1)
            A = adj_scales[scale]

            X = self.convs[layer_idx](X, A)
            X = self.norms[layer_idx](X)  # LayerNorm
            X = F.gelu(X)

            if layer_idx > 0:
                X = X + self.skip_cons[layer_idx - 1](residual)

            residual = X
            
            if self.use_attn_pool:
                pooled = self.pool(X)
            else:
                pooled = X.mean(dim=1)
            
            outs.append(pooled)

        # 返回特征序列，后续给 Mamba 做全局提取
        return [o.view(B, T_new, H) for o in outs]


class GraphTemporalFusion(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_heads=4, num_layers=3):
        super().__init__()
        # 这里的 GCN 已经是修改版了
        self.gcn = MultiScaleGCN(in_channels, hidden_dim, num_layers=num_layers,
                                 num_scales=3, use_attn_pool=True)
        
        # === 唯一需要改动的地方 ===
        # max_length: 6000 -> 300 (因为 CNN 把它变短了，不再需要那么大)
        # window_size: 保持 4 或者 8 都可以，现在 8 代表的时间跨度更大了，有助于全局理解
        self.cross_attn = CrossWindowSparseAttention(
            hidden_dim, 
            num_heads=num_heads,
            max_length=300,  # 适配压缩后的长度
            window_size=8,   
            num_neighbors=2  # 稍微减小邻居数，因为序列变短了
        )
        
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.Sigmoid()
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        # 输入 x: (B, T, C)
        # 调整为 (B, C, T) 给 GCN
        if x.shape[1] != 10 and x.shape[2] == 10:
             x = x.permute(0, 2, 1)
             
        gcn_outputs = self.gcn(x)  # List[(B, T_new, H)]
        
        shallow_feat = gcn_outputs[0]
        deep_feat = gcn_outputs[-1]
        
        # Cross Attention 正常工作，只是处理的序列变短了
        attn_shallow = self.cross_attn(
            query=shallow_feat,
            key=deep_feat,
            value=deep_feat
        )[0]
        
        attn_deep = self.cross_attn(
            query=deep_feat,
            key=shallow_feat,
            value=shallow_feat
        )[0]
        
        fused = self.fusion(torch.cat([attn_shallow, attn_deep], dim=-1))
        
        # 加上 Attention 结果
        fused = 0.3 * fused +   attn_deep
        
        return fused # (B, T_new, H) -> 传给 md
        
        
class MambaTemporalAggregator(nn.Module):
    """
    升级版：支持个体自适应 (Instance Adaptive)
    1. 使用 InstanceNorm 适应不同幅度的受试者
    2. 使用 动态路由 (Dynamic Routing) 根据当前信号自动调整 4 个方向的权重
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 预处理
        self.norm = nn.LayerNorm(hidden_dim)
        
        # ✅ 深度可分离卷积
        self.depthwise_conv = nn.Conv1d(
            hidden_dim, hidden_dim, 
            kernel_size=5, padding=2, groups=hidden_dim
        )
        self.pointwise_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        
        # [修改点 1] BatchNorm -> InstanceNorm (解决 Fold 8 低幅度问题)
        self.conv_norm = nn.InstanceNorm1d(hidden_dim, affine=True) 
        self.gelu = nn.GELU()
        
        # 池化
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        
        # GRU 定义保持不变
        gru_hidden = hidden_dim // 4 
        self.gru_forward = nn.GRU(hidden_dim, gru_hidden, batch_first=True)
        self.gru_backward = nn.GRU(hidden_dim, gru_hidden, batch_first=True)
        self.gru_vertical_down = nn.GRU(hidden_dim, gru_hidden, batch_first=True)
        self.gru_vertical_up = nn.GRU(hidden_dim, gru_hidden, batch_first=True)
        
        # 注意力池化
        self.attn_fc = nn.Linear(gru_hidden, 1)
        
        # [修改点 2] 静态参数 -> 动态路由网络
        # 原来是: self.direction_weights = nn.Parameter(...) 固定值
        # 现在是: 根据输入特征，动态生成 4 个权重
        self.weight_generator = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(), # Tanh 比 ReLU 在生成权重时更稳定
            nn.Linear(64, 4) # 输出 4 个方向的 logit
        )
        
        # 融合层
        total_dim = gru_hidden * 4
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # 残差投影
        self.residual_proj = nn.Linear(hidden_dim, hidden_dim)
        self.router_norm = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.25)
    
    def _attention_pooling(self, x):
        attn_scores = self.attn_fc(x)  # (B, T, 1)
        attn_weights = F.softmax(attn_scores, dim=1)
        pooled = (x * attn_weights).sum(dim=1)  # (B, D)
        return pooled
    
    def forward(self, x):
        B, T, D = x.shape
        x = torch.nan_to_num(x, nan=0.0)
        
        # 1. 预处理
        x_input = self.norm(x)
        
        # 2. 卷积 (BN -> IN)
        x_t = x_input.transpose(1, 2)
        x_conv = self.depthwise_conv(x_t)
        x_conv = self.pointwise_conv(x_conv)
        x_conv = self.gelu(self.conv_norm(x_conv)) # ✅ 使用 InstanceNorm
        
        # 3. 池化
        x_pool = self.pool(x_conv).transpose(1, 2)  # (B, T/2, D)
        
        # 4. 计算四个方向的特征
        # 正向
        gru_f, _ = self.gru_forward(x_pool)
        feat_f = self._attention_pooling(gru_f)
        
        # 反向
        gru_b, _ = self.gru_backward(torch.flip(x_pool, dims=[1]))
        feat_b = self._attention_pooling(gru_b)
        
        # 垂直下
        T_half = x_pool.size(1)
        indices = torch.arange(0, T_half, 2, device=x.device)
        if len(indices) * 2 < T_half:
            indices = torch.cat([indices, torch.arange(1, T_half, 2, device=x.device)])
        x_v = x_pool[:, indices, :]
        gru_vd, _ = self.gru_vertical_down(x_v)
        feat_vd = self._attention_pooling(gru_vd)
        
        # 垂直上
        gru_vu, _ = self.gru_vertical_up(torch.flip(x_v, dims=[1]))
        feat_vu = self._attention_pooling(gru_vu)
        
        # 5. [核心修改] 动态生成权重 (Content-Adaptive)
        # 我们利用 x_pool 的全局信息来决定要把重点放在哪个方向
        global_context = x_pool.mean(dim=1) # (B, D)
        # global_context = self.router_norm(global_context)
        dynamic_logits = self.weight_generator(global_context) # (B, 4)
        direction_w = F.softmax(dynamic_logits, dim=1) # (B, 4)
        
        # 6. 加权融合
        # Stack: (B, 4, D_sub)
        all_feats = torch.stack([feat_f, feat_b, feat_vd, feat_vu], dim=1)
        
        # Weighting: (B, 4, D_sub) * (B, 4, 1)
        weighted_feats = all_feats * direction_w.unsqueeze(-1)
        
        # 7. 拼接与最终融合
        concat_feats = weighted_feats.flatten(1)
        fused = self.fusion(concat_feats)
        
        # 残差
        residual = self.residual_proj(x_input.mean(dim=1))
        fused = fused + 0.2 * residual
        
        output = self.fc_out(fused)
        output = self.dropout(output)
        
        return output
