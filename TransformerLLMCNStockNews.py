import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, roc_auc_score
from datetime import datetime
import logging
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch.nn.utils as utils
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm  # 进度条
import ta  # 技术分析库
from transformers import BertTokenizer, BertModel
from collections import defaultdict
import json  # 确保导入 json 模块
from transformers import RobertaTokenizer, RobertaModel, DebertaV2Tokenizer, DebertaV2Model
from transformers import RobertaTokenizer, RobertaForMaskedLM, DebertaV2Tokenizer, DebertaV2ForMaskedLM
# 设置日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 辅助函数：创建滑动窗口（使用生成器）
def create_sliding_window(X, window_size):
    """使用生成器逐批次生成滑动窗口"""
    num_samples = len(X) - window_size + 1
    if num_samples <= 0:
        return None
    
    logging.info(f"Creating sliding windows with window_size={window_size}, num_samples={num_samples}")
    
    for i in range(num_samples):
        yield X[i:i + window_size]

# 添加技术指标作为新特征
def add_technical_indicators(df):
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    df['BB_upper'] = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2).bollinger_hband()
    df['BB_lower'] = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2).bollinger_lband()
    return df.dropna()  # 移除新增特征后的 NaN 值

# 定义 Transformer 模型架构
class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, output_size, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        # 定义 Transformer 编码器
        self.transformer = nn.Transformer(
            d_model=hidden_size,  # 模型的隐藏层大小
            nhead=num_heads,      # 多头注意力机制的头数
            num_encoder_layers=num_layers,  # 编码器层数
            num_decoder_layers=num_layers,  # 解码器层数
            dim_feedforward=hidden_size * 4,  # 前馈网络的维度
            dropout=dropout,  # Dropout 概率
            batch_first=True  # 批量优先
        )
        
        # 定义输入嵌入层
        self.embedding = nn.Linear(input_size, hidden_size)
        
        # 定义输出层
        self.fc_out = nn.Linear(hidden_size, output_size)
        
        # 初始化权重
        self.apply(self.init_weights)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x_enc, x_dec):
        # 将输入特征映射到隐藏层大小
        x_enc = self.embedding(x_enc)
        x_dec = self.embedding(x_dec)
        
        # 通过 Transformer 编码器-解码器
        output = self.transformer(src=x_enc, tgt=x_dec)
        
        # 通过输出层
        output = self.fc_out(output[:, -1, :])  # 只取最后一个时间步的输出
        
        return output

# 读取股票数据
def read_stock_data(stock_folder_path):
    stock_data = {}
    for i, stock_file in enumerate(os.listdir(stock_folder_path)):
        if stock_file.endswith('.txt'):
            company_name = stock_file.split('.')[0]  # 获取公司名称
            file_path = os.path.join(stock_folder_path, stock_file)
            
            # 打印文件路径，确保文件存在
            logging.info(f"Reading file {i + 1}/{len(os.listdir(stock_folder_path))}: {file_path}")
            
            try:
                df = pd.read_csv(file_path, sep='\t', header=None, names=["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"])
                df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
                
                # 检查是否有缺失值
                if df.isnull().values.any():
                    logging.warning(f"Company {company_name} has missing values in the data.")
                    continue
                
                # 使用 company_name 作为键，而不是 company
                stock_data[company_name] = df
            except Exception as e:
                logging.error(f"Error reading file {file_path}: {e}")
    
    return stock_data

# 读取新闻数据
def read_news_data(news_folder_path):
    news_data = defaultdict(list)
    for company_folder in os.listdir(news_folder_path):
        company_folder_path = os.path.join(news_folder_path, company_folder)
        if os.path.isdir(company_folder_path):  # 确保是目录
            for date_file in os.listdir(company_folder_path):
                date_file_path = os.path.join(company_folder_path, date_file)
                try:
                    with open(date_file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                entry = json.loads(line.strip())
                                text = entry.get('text', '')
                                created_at = entry.get('created_at', '')
                                if text and created_at:
                                    created_at = datetime.strptime(created_at, '%Y-%m-%d').strftime('%Y-%m-%d')
                                    news_data[company_folder].append({'text': text, 'created_at': created_at})
                            except json.JSONDecodeError as e:
                                logging.error(f"Error decoding JSON line in file {date_file_path}: {e}")
                except Exception as e:
                    logging.error(f"Error reading file {date_file_path}: {e}")
    
    return news_data

# 使用 BERT 对新闻进行嵌入
# 使用Roberta对新闻进行嵌入
def get_roberta_embedding(texts, batch_size=10):
    texts = [text for text in texts if isinstance(text, str) and len(text.strip()) > 0]
    if not texts:
        logging.warning("No valid texts to embed.")
        return []

    logging.info(f"Embedding {len(texts)} texts.")
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertModel.from_pretrained('bert-base-chinese').to(device)
    # 加载Roberta模型和分词器
    # tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    # model = RobertaModel.from_pretrained('roberta-base').to(device)

    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # 这里假设取[CLS]标记的输出作为文本嵌入，可能需要根据实际情况调整
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.extend(cls_embeddings)

    return embeddings

# 聚合新闻和股票数据以适应 Transformer
def aggregate_data_for_transformer(stock_data, news_data, bert_embeddings, window_size=32):
    aggregated_data = {}
    for company_name, stock_df in stock_data.items():
        logging.info(f"Processing company: {company_name}")
        
        # 添加技术指标
        stock_df = add_technical_indicators(stock_df)
        
        # 确保日期是字符串格式
        stock_df['Date'] = stock_df['Date'].dt.strftime('%Y-%m-%d')
        
        # 创建特征和标签
        merged_data = []
        dates = []  # 保存日期信息
        labels = []
        news_embeddings = []
        
        for i in range(len(stock_df) - 1):  # 使用前一天的数据预测后一天的价格变化
            date = stock_df.iloc[i]['Date']
            stock_features = stock_df.iloc[i][['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_10', 'RSI', 'BB_upper', 'BB_lower']].values.astype(np.float32)  # 强制转换为 float32
            merged_data.append(stock_features)
            dates.append(date)
            
            # 生成标签：如果后一天的收盘价高于当天，则标签为 1，否则为 0
            next_day_close = stock_df.iloc[i + 1]['Close']
            label = 1 if next_day_close > stock_df.iloc[i]['Close'] else 0
            labels.append(label)
            
            # 查找当天的新闻嵌入
            if company_name in bert_embeddings and date in bert_embeddings[company_name]:
                news_embeddings.append(bert_embeddings[company_name][date])
            else:
                # 如果没有当天的新闻嵌入，使用零向量填充
                news_embeddings.append(np.zeros(768))
        
        if merged_data and labels:
            if len(dates) != len(labels):
                logging.error(f"Length mismatch for company {company_name}: len(dates)={len(dates)}, len(labels)={len(labels)}")
                continue
            
            # 归一化输入数据
            scaler = MinMaxScaler()
            normalized_stock_data = scaler.fit_transform(merged_data)
            
            # 将新闻嵌入与股票特征拼接
            combined_data = np.concatenate([normalized_stock_data, np.array(news_embeddings)], axis=1)
            
            aggregated_data[company_name] = {
                'data': combined_data,  # 归一化后的数据 + 新闻嵌入
                'dates': dates,  # 保存日期信息
                'labels': np.array(labels, dtype=np.int64)  # 强制转换为 int64
            }
        else:
            logging.warning(f"No valid data for company {company_name}. Skipping...")
    
    return aggregated_data

# 训练 Transformer 模型
def train_transformer_model(aggregated_data, test_start_date='2021-01-03'):
    mcc_list = []
    acc_list = []
    f1_list = []
    roc_auc_list = []

    for i, (company_name, data) in enumerate(aggregated_data.items()):
        logging.info(f"Training model for company {i + 1}/{len(aggregated_data)}: {company_name}")
        
        X = data['data']
        y_true = data['labels']
        dates = data['dates']  # 使用保存的日期信息
        
        # 确保 dates 和 y_true 的长度一致
        if len(dates) != len(y_true):
            logging.error(f"Length mismatch for company {company_name}: len(dates)={len(dates)}, len(labels)={len(y_true)}")
            continue
        
        # 转换为 numpy 数组
        X = np.array(X, dtype=np.float32)
        y_true = np.array(y_true, dtype=np.int64)
        dates = [datetime.strptime(date, '%Y-%m-%d') for date in dates]  # 将日期字符串转换为 datetime 对象
        
        # 确保数据按日期排序（如果尚未排序）
        sorted_indices = np.argsort(dates)
        X = X[sorted_indices]
        y_true = y_true[sorted_indices]
        dates = [dates[i] for i in sorted_indices]  # 也对日期进行排序
        
        # 根据 test_start_date 分割训练集和测试集
        test_start_idx = next((j for j, date in enumerate(dates) if date >= datetime.strptime(test_start_date, '%Y-%m-%d')), None)
        if test_start_idx is None:
            logging.warning(f"No test data for company {company_name}. Skipping...")
            continue
        
        X_train, X_test = X[:test_start_idx], X[test_start_idx:]
        y_train, y_test = y_true[:test_start_idx], y_true[test_start_idx:]
        
        if len(X_train) == 0 or len(X_test) == 0:
            logging.warning(f"Not enough data for company {company_name}. Skipping...")
            continue
        
        # 计算 window_size 和 num_features
        window_size = min(32, len(X_train) - 1)  # 确保 window_size 不超过训练数据的长度减去 1
        num_features = X_train.shape[1]  # 每个时间步的特征数量
        
        # 创建滑动窗口生成器
        X_train_windows = list(create_sliding_window(X_train, window_size))
        X_test_windows = list(create_sliding_window(X_test, window_size))
        
        if not X_train_windows or not X_test_windows:
            logging.warning(f"Not enough data for company {company_name}. Skipping...")
            continue
        
        # 计算正负样本的比例，用于加权损失函数
        pos_weight = torch.tensor([len(y_train) / sum(y_train)], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # 初始化 Transformer 模型
        model = TransformerModel(
            input_size=num_features,
            hidden_size=64,  # 增加隐藏层大小
            num_layers=2,   # 增加编码器和解码器的层数
            num_heads=8,    # 增加多头注意力机制的头数
            output_size=1,  # 输出维度（二分类任务）
            dropout=0.2     # 增加 Dropout 概率
        ).to(device)
        
        # 定义优化器
        optimizer = optim.Adam(model.parameters(), lr=1e-4)  # 适当调整学习率
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)  # 动态调整学习率
        
        # 使用 DataLoader 实现批量训练
        train_dataset = TensorDataset(torch.tensor(X_train_windows, dtype=torch.float32), torch.tensor(y_train[window_size - 1:], dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # 减小批量大小
        
        # 训练模型
        model.train()
        best_loss = float('inf')
        early_stopping_patience = 5
        early_stopping_counter = 0
        
        for epoch in range(20):  # 增加训练轮数
            epoch_loss = 0.0
            for batch_idx, (x_enc, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
                x_enc = x_enc.to(device)
                y = y.unsqueeze(1).to(device)  # 增加一个维度
                
                # 前向传播
                outputs = model(x_enc, x_enc[:, -1:, :])  # 使用最后一个时间步作为解码器输入
                loss = criterion(outputs, y)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            logging.info(f"Epoch {epoch + 1}, Average Loss: {avg_epoch_loss:.4f}")
            
            # 更新学习率
            scheduler.step(avg_epoch_loss)
            
            # 早停机制
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    logging.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break
        
        # 评估模型
        model.eval()
        test_dataset = TensorDataset(torch.tensor(X_test_windows, dtype=torch.float32), torch.tensor(y_test[window_size - 1:], dtype=torch.float32))
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)  # 使用批量评估
        
        y_pred_prob = []
        with torch.no_grad():
            for x_enc, y in tqdm(test_loader, desc="Evaluating"):
                x_enc = x_enc.to(device)
                outputs = model(x_enc, x_enc[:, -1:, :])
                y_pred_prob.extend(outputs.squeeze().cpu().numpy())
        
        y_pred_prob = np.array(y_pred_prob)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        mcc = matthews_corrcoef(y_test[window_size - 1:], y_pred)
        acc = accuracy_score(y_test[window_size - 1:], y_pred)
        f1 = f1_score(y_test[window_size - 1:], y_pred)
        roc_auc = roc_auc_score(y_test[window_size - 1:], y_pred_prob)
        
        mcc_list.append(mcc)
        acc_list.append(acc)
        f1_list.append(f1)
        roc_auc_list.append(roc_auc)
        
        # 手动释放显存
        del model, optimizer, y_pred, y_pred_prob
        torch.cuda.empty_cache()
    
    if len(mcc_list) == 0 or len(acc_list) == 0 or len(f1_list) == 0 or len(roc_auc_list) == 0:
        print("No valid data for evaluation.")
    else:
        print(f'MCC: {sum(mcc_list) / len(mcc_list):.4f}')
        print(f'ACC: {sum(acc_list) / len(acc_list):.4f}')
        print(f'F1 Score: {sum(f1_list) / len(f1_list):.4f}')
        print(f'ROC AUC: {sum(roc_auc_list) / len(roc_auc_list):.4f}')

# 主函数
def main(stock_folder_path, news_folder_path):
    # 读取股票数据
    stock_data = read_stock_data(stock_folder_path)
    
    # 读取新闻数据
    news_data = read_news_data(news_folder_path)
    
    # 使用 BERT 对新闻进行嵌入
    bert_embeddings = defaultdict(dict)
    for company_name, articles in news_data.items():
        valid_texts = [' '.join(article['text']) for article in articles if isinstance(article.get('text'), list) and article['text'] and all(isinstance(word, str) for word in article['text'])]
        if not valid_texts:
            logging.warning(f"No valid news articles for company {company_name}. Skipping...")
            continue
        
        embeddings = get_roberta_embedding(valid_texts)
        for i, article in enumerate(articles):
            date = article['created_at']
            if i < len(embeddings):
                bert_embeddings[company_name][date] = embeddings[i]
            else:
                logging.warning(f"Missing embedding for article on {date} for company {company_name}. Using zero vector.")
                bert_embeddings[company_name][date] = np.zeros(768)
    
    # 聚合新闻和股票数据以适应 Transformer
    aggregated_data = aggregate_data_for_transformer(stock_data, news_data, bert_embeddings)
    
    # 训练 Transformer 模型
    train_transformer_model(aggregated_data, test_start_date='2021-01-03')

if __name__ == "__main__":
    stock_folder_path = "CMIN-Dataset-main/CMIN-CN/price/preprocessed"  # 替换为实际的股票数据路径
    news_folder_path = "CMIN-Dataset-main/CMIN-CN/news/preprocessed"  # 替换为实际的新闻数据路径
    main(stock_folder_path, news_folder_path)