import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from collections import Counter
import re

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 数据预处理 - 只使用1%的数据
class TextDataset(Dataset):
    def __init__(self, text, seq_length=64, use_percentage=0.01):
        self.text = text
        self.seq_length = seq_length

        # 只使用前1%的数据
        total_length = len(text)
        self.text = text[:int(total_length * use_percentage)]

        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

        # 将文本转换为索引
        self.data = [self.char_to_idx[ch] for ch in self.text]

        print(f"使用数据量: {len(self.text)} 字符")
        print(f"词汇表大小: {self.vocab_size}")

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx + self.seq_length], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1:idx + self.seq_length + 1], dtype=torch.long)
        return x, y



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].transpose(0, 1)


#  多头自注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len = query.size(0), query.size(1)

        # 线性变换并分头
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # 应用注意力权重
        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.w_o(out)


# 位置前馈网络
class PositionwiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFFN, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


#  编码器层
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 自注意力 + 残差连接 + 层归一化
        attn_out = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # 前馈网络 + 残差连接 + 层归一化
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x


#  解码器层
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # 掩码自注意力
        attn1 = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn1))

        # 交叉注意力
        attn2 = self.cross_attention(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn2))

        # 前馈网络
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))

        return x


#  编码器
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        return x


#  解码器
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.linear = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)

        return self.linear(x)


# 完整 Transformer
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_layers=6,
                 num_heads=8, d_ff=2048, max_seq_length=5000, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length, dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        return dec_output


#  计算准确率的函数
def calculate_accuracy(output, target):
    """计算预测准确率"""
    _, predicted = torch.max(output, dim=-1)
    correct = (predicted == target).float()
    accuracy = correct.sum() / correct.numel()
    return accuracy.item()


#  训练函数
def train_transformer():
    # 读取数据
    with open('tiny_shakespeare.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"原始文本长度: {len(text)}")
    print(f"前200个字符: {text[:200]}")

    # 创建数据集 - 只使用1%的数据
    dataset = TextDataset(text, seq_length=64, use_percentage=0.01)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    print(f"训练样本数量: {len(dataset)}")

    # 初始化更小的模型
    model = Transformer(
        src_vocab_size=dataset.vocab_size,
        tgt_vocab_size=dataset.vocab_size,
        d_model=64,
        num_layers=2,
        num_heads=2,
        d_ff=256,
        max_seq_length=5000,
        dropout=0.1
    )

    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # 训练循环 - 记录损失和准确率
    losses = []
    accuracies = []
    for epoch in range(15):
        total_loss = 0
        total_accuracy = 0
        model.train()

        for batch_idx, (src, tgt) in enumerate(dataloader):
            optimizer.zero_grad()

            # 准备输入输出
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            # 前向传播
            output = model(src, tgt_input)

            # 计算损失和准确率
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
            accuracy = calculate_accuracy(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_accuracy += accuracy

            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

        avg_loss = total_loss / len(dataloader)
        avg_accuracy = total_accuracy / len(dataloader)
        losses.append(avg_loss)
        accuracies.append(avg_accuracy)
        scheduler.step()

        print(f'Epoch {epoch} completed. Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_accuracy:.4f}')

        # 生成示例文本
        if epoch % 3 == 0:
            generate_text(model, dataset, "First Citizen:", max_length=30)

    return model, dataset, losses, accuracies


def generate_text(model, dataset, start_text, max_length=100):
    model.eval()
    with torch.no_grad():
        # 将起始文本转换为索引
        input_ids = [dataset.char_to_idx.get(ch, 0) for ch in start_text]
        generated = input_ids.copy()

        for _ in range(max_length):
            src = torch.tensor([input_ids], dtype=torch.long)
            tgt = torch.tensor([generated], dtype=torch.long)

            output = model(src, tgt[:, :-1])
            next_token_logits = output[0, -1, :]
            next_token = torch.argmax(next_token_logits).item()

            generated.append(next_token)

            # 简单的停止条件
            if len(generated) > len(start_text) + max_length:
                break

        # 将索引转换回文本
        generated_text = ''.join([dataset.idx_to_char.get(idx, '?') for idx in generated])
        print(f"生成文本: {generated_text}")

    return generated_text


#  消融实验
def ablation_study():
    with open('tiny_shakespeare.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # 只使用1%的数据
    dataset = TextDataset(text, seq_length=64, use_percentage=0.01)

    # 不同配置的实验
    configs = [
        {'name': 'Baseline', 'num_layers': 2, 'num_heads': 2, 'd_model': 64, 'd_ff': 256},
        {'name': 'Small_Model', 'num_layers': 1, 'num_heads': 1, 'd_model': 32, 'd_ff': 128},
        {'name': 'No_FFN', 'num_layers': 2, 'num_heads': 2, 'd_model': 64, 'd_ff': 64, 'no_ffn': True},
        {'name': 'Single_Head', 'num_layers': 2, 'num_heads': 1, 'd_model': 64, 'd_ff': 256},
        {'name': 'No_Positional', 'num_layers': 2, 'num_heads': 2, 'd_model': 64, 'd_ff': 256, 'no_pos': True},
    ]

    results = {}

    for config in configs:
        print(f"\n训练 {config['name']}...")

        class CustomTransformer(Transformer):
            def __init__(self, *args, **kwargs):
                self.no_ffn = kwargs.pop('no_ffn', False)
                self.no_pos = kwargs.pop('no_pos', False)
                super().__init__(*args, **kwargs)

                if self.no_ffn:
                    # 移除FFN层
                    for layer in self.encoder.layers:
                        layer.ffn = nn.Identity()
                    for layer in self.decoder.layers:
                        layer.ffn = nn.Identity()

                if self.no_pos:
                    # 移除位置编码
                    self.encoder.pos_encoding = nn.Identity()
                    self.decoder.pos_encoding = nn.Identity()

        model = CustomTransformer(
            src_vocab_size=dataset.vocab_size,
            tgt_vocab_size=dataset.vocab_size,
            d_model=config['d_model'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            d_ff=config['d_ff'],
            no_ffn=config.get('no_ffn', False),
            no_pos=config.get('no_pos', False)
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

        losses = []
        accuracies = []
        for epoch in range(5):  # 增加训练轮数以获得更准确的结果
            total_loss = 0
            total_accuracy = 0
            model.train()

            for src, tgt in dataloader:
                optimizer.zero_grad()

                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                output = model(src, tgt_input)
                loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
                accuracy = calculate_accuracy(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                total_accuracy += accuracy

            avg_loss = total_loss / len(dataloader)
            avg_accuracy = total_accuracy / len(dataloader)
            losses.append(avg_loss)
            accuracies.append(avg_accuracy)

        results[config['name']] = {
            'losses': losses,
            'accuracies': accuracies,
            'final_accuracy': accuracies[-1]
        }
        print(f"{config['name']} 最终损失: {avg_loss:.4f}, 最终准确率: {avg_accuracy:.4f}")

        # 为每个模型生成一个示例文本
        generate_text(model, dataset, "First Citizen:", max_length=20)

    return results


#  可视化结果
def plot_results(losses, accuracies, ablation_results):
    plt.figure(figsize=(20, 10))

    # 主训练损失和准确率
    plt.subplot(2, 3, 1)
    plt.plot(losses)
    plt.title('训练损失 (1% 数据)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.subplot(2, 3, 2)
    plt.plot(accuracies)
    plt.title('训练准确率 (1% 数据)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)

    # 消融实验损失比较
    plt.subplot(2, 3, 3)
    for name, result in ablation_results.items():
        plt.plot(result['losses'], label=name)
    plt.title('消融实验 - 训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 消融实验准确率比较
    plt.subplot(2, 3, 4)
    for name, result in ablation_results.items():
        plt.plot(result['accuracies'], label=name)
    plt.title('消融实验 - 训练准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # 最终准确率对比柱状图
    plt.subplot(2, 3, 5)
    names = list(ablation_results.keys())
    final_accuracies = [ablation_results[name]['final_accuracy'] for name in names]
    colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
    bars = plt.bar(names, final_accuracies, color=colors)
    plt.title('消融实验 - 最终准确率对比')
    plt.ylabel('Final Accuracy')
    plt.xticks(rotation=45)

    # 在柱状图上添加数值标签
    for bar, accuracy in zip(bars, final_accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                 f'{accuracy:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('transformer_results_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.show()


#  结果
def analyze_results(model, dataset, ablation_results):
    print("\n=== 结果分析 ===")

    # 计算困惑度
    model.eval()
    with torch.no_grad():
        test_loader = DataLoader(dataset, batch_size=4, shuffle=False)
        total_loss = 0
        total_accuracy = 0
        criterion = nn.CrossEntropyLoss()

        for src, tgt in test_loader:
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            output = model(src, tgt_input)
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
            accuracy = calculate_accuracy(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))

            total_loss += loss.item()
            total_accuracy += accuracy

        avg_loss = total_loss / len(test_loader)
        avg_accuracy = total_accuracy / len(test_loader)
        perplexity = math.exp(avg_loss)
        print(f"测试集平均损失: {avg_loss:.4f}")
        print(f"测试集平均准确率: {avg_accuracy:.4f}")
        print(f"困惑度: {perplexity:.2f}")

    # 消融实验分析
    print("\n=== 消融实验分析 ===")
    for name, result in ablation_results.items():
        print(f"{name}: 最终准确率 = {result['final_accuracy']:.4f}")

    # 生成更多示例
    print("\n=== 生成文本示例 ===")
    prompts = ["First Citizen:", "MENENIUS:", "The gods", "I pray"]

    for prompt in prompts:
        generated = generate_text(model, dataset, prompt, max_length=30)
        print(f"提示: '{prompt}' -> 生成: '{generated}'\n")


# 主执行函数
if __name__ == "__main__":
    print("开始 Transformer 训练 (使用1%数据)...")

    try:
        # 主训练
        model, dataset, losses, accuracies = train_transformer()

        # 消融实验
        ablation_results = ablation_study()

        # 可视化结果
        plot_results(losses, accuracies, ablation_results)

        # 结果
        analyze_results(model, dataset, ablation_results)

        # 保存模型
        torch.save({
            'model_state_dict': model.state_dict(),
            'dataset_info': {
                'char_to_idx': dataset.char_to_idx,
                'idx_to_char': dataset.idx_to_char,
                'vocab_size': dataset.vocab_size
            }
        }, 'transformer_model_1percent.pth')

        print("训练完成!")

    except Exception as e:
        print(f"训练过程中出错: {e}")
        import traceback

        traceback.print_exc()
