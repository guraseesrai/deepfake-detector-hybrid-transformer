class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        # Convolutional and residual layers remain unchanged
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(3, 7, 7),
                               stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.LeakyReLU(0.1)
        self.pool = nn.MaxPool3d(kernel_size=3, stride=(1, 2, 2), padding=1)
        self.layer1 = self.make_layer(64, 64, num_blocks=3, stride=1)
        self.layer2 = self.make_layer(64 * Bottleneck.expansion, 128, num_blocks=4, stride=2)
        self.layer3 = self.make_layer(128 * Bottleneck.expansion, 256, num_blocks=6, stride=2)
        self.layer4 = self.make_layer(256 * Bottleneck.expansion, 512, num_blocks=3, stride=2)

        # Transformer parameters
        self.d_model = 512 * Bottleneck.expansion  # typically 2048 in your case
        # Learnable classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        # Positional embedding: initialized later based on sequence length
        self.pos_embedding = None

        # Transformer Encoder Layer and Encoder
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=2)
        # Final classification layer
        self.fc = nn.Linear(self.d_model, 1)

    def make_layer(self, in_channels, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels * Bottleneck.expansion)
            )

        layers = [Bottleneck(in_channels, out_channels, stride, downsample)]
        for _ in range(1, num_blocks):
            layers.append(Bottleneck(out_channels * Bottleneck.expansion, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Flatten the spatio-temporal dimensions: [B, C, T, H, W] -> [B, N, C]
        B, C, T, H, W = x.shape
        x = x.view(B, C, -1)  # N = T * H * W
        x = x.permute(0, 2, 1)  # shape: [B, N, C]

        # Prepend the classification token for each sample
        cls_tokens = self.cls_token.expand(B, -1, -1)  # shape: [B, 1, C]
        x = torch.cat((cls_tokens, x), dim=1)  # new shape: [B, N+1, C]

        # Create and add positional embeddings (if not already initialized or if shape differs)
        seq_length = x.shape[1]
        if self.pos_embedding is None or self.pos_embedding.shape[1] != seq_length:
            self.pos_embedding = nn.Parameter(torch.randn(1, seq_length, C, device=x.device))
        x = x + self.pos_embedding

        # Pass through the transformer encoder
        x = self.transformer_encoder(x)
        # Use the output of the CLS token as the aggregated representation
        x = x[:, 0]  # shape: [B, C]

        x = self.fc(x)  # shape: [B, 1]
        return x
