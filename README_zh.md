# 超級瑪莉兄弟強化學習代理

## 資料夾結構

- **DQN.py**: 包含深度 Q 網路 (DQN) 和 Actor-Critic DQN (ACDQN) 演算法的實現。
- **eval.py**: 用於評估訓練模型的性能。
- **find_best_model.py**: 測試不同的跳幀值以找到最佳模型的腳本。
- **model.py**: 定義 DQN 的神經網路架構。
- **model_AC.py**: 定義 Actor-Critic 模型的神經網路架構。
- **reward.py**: 包含多種獎勵函數，用於激勵代理的行為。
- **run.py**: 強化學習代理的主要訓練迴圈。
- **skipframe.py**: 用於跳過訓練過程中的幀以加快速度的包裝器。
- **movement.py**: 定義代理的可能動作/行為。
- **utils.py**: 包含實用工具函數，例如幀的預處理。

## 安裝

### 複製存儲庫：

```bash
git clone https://github.com/livejiaquan/Super-mario-bros.git
cd Super-mario-bros
```

### 安裝所需依賴項：

```bash
pip install -r requirements.txt
```

## 使用方法

### 訓練

要訓練強化學習代理，運行 `run.py` 腳本。此腳本會初始化環境，設置 DQN 或 ACDQN 代理，並開始訓練過程。

```bash
python run.py
```

### 評估

要評估訓練好的模型，使用 `eval.py` 腳本。請確保在腳本中指定模型權重的路徑。

```bash
python eval.py
```

### 找到最佳模型

要通過測試不同的跳幀值找到最佳模型，運行 `find_best_model.py` 腳本。

```bash
python find_best_model.py
```

## 獎勵函數

`reward.py` 文件包含多種函數，用於根據代理的行為和遊戲的狀態計算獎勵。這些函數包括收集硬幣、進行垂直和水平移動、保持速度等的獎勵。

## 自訂動作

`movement.py` 文件定義了代理可以執行的不同動作/行為集合。`CUSTOM_MOVEMENT` 集合包括向右移動、跳躍和奔跑等行為。

## 預處理

`utils.py` 文件包含將遊戲幀預處理為灰階並調整為 84x84 像素的函數。

## 跳幀包裝器

`skipframe.py` 文件定義了一個包裝器，用於在訓練過程中跳過幀，這有助於通過減少代理需要處理的幀數來加快訓練過程。

## 模型架構

- **model.py**: 包含使用卷積和殘差塊的 DQN 模型架構。
- **model_AC.py**: 包含共享卷積層以及獨立 actor 和 critic 頭部的 Actor-Critic 模型架構。

## 授權

此專案採用 MIT 授權條款。

## 致謝

本專案使用了 `gym_super_mario_bros` 環境來模擬超級瑪莉兄弟遊戲，以訓練強化學習代理。特別感謝使用本專案所需的各類庫和工具的作者。

更多信息請參考存儲庫中的相關文件。
