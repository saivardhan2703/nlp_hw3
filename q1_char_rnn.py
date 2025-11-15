import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ----------------------------------------------------
# 1. LOAD TEXT
# ----------------------------------------------------
with open("data/text.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
stoi = {c:i for i,c in enumerate(chars)}
itos = {i:c for c,i in stoi.items()}
vocab_size = len(chars)

print("Vocab size:", vocab_size)

# Encode entire text as integers
encoded = torch.tensor([stoi[c] for c in text], dtype=torch.long)

# ----------------------------------------------------
# 2. DATASET
# ----------------------------------------------------
class CharDataset(Dataset):
    def __init__(self, data, seq_len=100):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+1:idx+self.seq_len+1]
        return x, y

dataset = CharDataset(encoded, seq_len=100)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# ----------------------------------------------------
# 3. MODEL
# ----------------------------------------------------
class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x, h=None):
        x = self.embed(x)
        out, h = self.rnn(x, h)
        return self.fc(out), h

model = CharRNN(vocab_size, 128, 256)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ----------------------------------------------------
# 4. TRAIN
# ----------------------------------------------------
for epoch in range(5):
    total_loss = 0
    for x, y in dataloader:
        pred, _ = model(x)
        loss = criterion(pred.view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss = {total_loss/len(dataloader):.4f}")

# ----------------------------------------------------
# 5. SAMPLING FUNCTION
# ----------------------------------------------------
def sample(model, start='H', temp=1.0, length=300):
    model.eval()
    inp = torch.tensor([[stoi[start]]])
    out_text = start
    hidden = None

    for _ in range(length):
        logits, hidden = model(inp, hidden)
        logits = logits[:, -1, :] / temp
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        out_text += itos[next_id]
        inp = torch.tensor([[next_id]])

    return out_text

# ----------------------------------------------------
# 6. GENERATE SAMPLES
# ----------------------------------------------------
print("\nSample (temp=0.7):")
print(sample(model, temp=0.7))

print("\nSample (temp=1.0):")
print(sample(model, temp=1.0))

print("\nSample (temp=1.2):")
print(sample(model, temp=1.2))
