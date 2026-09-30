import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

class MiniLM_BiGRU_V4(nn.Module):
    def __init__(self, num_classes=6, unfreeze_layers=2):
        super().__init__()
        self.transformer = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        
        for param in self.transformer.parameters():
            param.requires_grad = False
            
        if unfreeze_layers > 0:
            for layer in self.transformer.encoder.layer[-unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        self.bigru = nn.GRU(
            input_size=384, 
            hidden_size=64, 
            batch_first=True, 
            bidirectional=True
        )
        self.dropout1 = nn.Dropout(0.3)
        self.dense1 = nn.Linear(128, 32)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.dense2 = nn.Linear(32, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        mask = attention_mask.unsqueeze(-1).float()
        hidden_states = hidden_states * mask
        
        lengths = attention_mask.sum(dim=1).cpu()
        lengths = lengths.clamp(min=1)
        
        packed_input = nn.utils.rnn.pack_padded_sequence(hidden_states, lengths, batch_first=True, enforce_sorted=False)
        _, h_n = self.bigru(packed_input)
        
        gru_out = torch.cat([h_n[0], h_n[1]], dim=1)
        
        x = self.dropout1(gru_out)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.dense2(x)
        return x

def preprocess_text(text: str) -> str:
    import re
    return re.sub(r'\s+', ' ', text).strip()

def prepare_dataloaders(batch_size=32):
    from datasets import load_dataset
    dataset = load_dataset("dair-ai/emotion")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
    def process_split(split):
        texts = [preprocess_text(x) for x in split["text"]]
        labels = split["label"]
        encoded = tokenizer(texts, max_length=50, padding="max_length", truncation=True, return_tensors="pt")
        return TensorDataset(encoded["input_ids"], encoded["attention_mask"], torch.tensor(labels, dtype=torch.long))

    train_ds = process_split(dataset["train"])
    val_ds = process_split(dataset["validation"])
    
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(val_ds, batch_size=batch_size, shuffle=False), dataset["train"]["label"]

def main():
    from sklearn.metrics import f1_score
    from sklearn.utils.class_weight import compute_class_weight
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader, val_loader, train_labels = prepare_dataloaders(batch_size=32)
    
    model = MiniLM_BiGRU_V4(unfreeze_layers=1).to(device)
    
    classes = np.unique(train_labels)
    # Conservative class weights: blend balanced with uniform to prevent majority collapse
    # "Do NOT over-weight the minority classes so aggressively that the majority-class performance collapses."
    weights = compute_class_weight("balanced", classes=classes, y=train_labels)
    conservative_weights = (weights + 1.0) / 2.0
    
    weight_tensor = torch.ones(6, dtype=torch.float32)
    for c, w in zip(classes, conservative_weights):
        weight_tensor[c] = float(w)
    weight_tensor = weight_tensor.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    
    transformer_params = [p for n, p in model.transformer.named_parameters() if p.requires_grad]
    classifier_params = list(model.bigru.parameters()) + list(model.dense1.parameters()) + list(model.dense2.parameters())
    
    optimizer = torch.optim.AdamW([
        {'params': transformer_params, 'lr': 2e-5},
        {'params': classifier_params, 'lr': 1e-3}
    ])
    
    epochs = 6
    patience = 2
    best_f1 = -1.0
    patience_counter = 0
    save_path = "Artifacts/MiniLM_Sequence_Classifier_v4.pt"
    os.makedirs("Artifacts", exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for b_id, (ids, mask, labels) in enumerate(train_loader):
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(ids, mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if b_id % 50 == 0:
                print(f"Epoch {epoch+1} Batch {b_id}/{len(train_loader)} Loss: {loss.item():.4f}")
            
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for ids, mask, labels in val_loader:
                ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
                outputs = model(ids, mask)
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(labels.cpu().numpy())
                
        macro_f1 = f1_score(val_true, val_preds, average="macro")
        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f} Val Macro F1: {macro_f1:.4f}")
        
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved new best model to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

if __name__ == '__main__':
    main()
