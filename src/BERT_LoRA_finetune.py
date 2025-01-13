from transformers import AutoModel, AutoTokenizer, PreTrainedModel
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Union
import math

class LoRALinear(nn.Module):
    """
    Implementation of LoRA-augmented Linear layer.
    Instead of learning the full weight matrix, we learn two low-rank matrices A and B.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 1.0,
        freeze_weights: bool = True
    ):
        super().__init__()
        # Original linear layer
        self.linear = nn.Linear(in_features, out_features)
        if freeze_weights:
            self.linear.weight.requires_grad = False
            if self.linear.bias is not None:
                self.linear.bias.requires_grad = False
        
        self.rank = rank
        self.alpha = alpha
        scaling = alpha / rank
        
        # LoRA matrices
        # Note the dimension order: we want (in_features × rank) × (rank × out_features)
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) / math.sqrt(rank))
        # Initialize B to zero so LoRA starts as identity mapping
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scaling = scaling
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original transformation
        base_output = self.linear(x)
        
        # LoRA transformation
        lora_output = (x @ self.lora_A @ self.lora_B) * self.scaling
        
        return base_output + lora_output

class LoRABERTClassifier(nn.Module):
    """
    BERT classifier with LoRA adaptations applied to attention layers.
    """
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_classes: int = 2,
        lora_rank: int = 8,
        lora_alpha: float = 1.0,
        freeze_bert: bool = True
    ):
        super().__init__()
        # Load pretrained BERT
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Replace attention query and value projections with LoRA versions
        # This is where we'll concentrate our adaptations
        hidden_size = self.bert.config.hidden_size
        
        for layer in self.bert.encoder.layer:
            # Replace query projection
            layer.attention.self.query = LoRALinear(
                hidden_size,
                hidden_size,
                rank=lora_rank,
                alpha=lora_alpha,
                freeze_weights=freeze_bert
            )
            # Replace value projection
            layer.attention.self.value = LoRALinear(
                hidden_size,
                hidden_size,
                rank=lora_rank,
                alpha=lora_alpha,
                freeze_weights=freeze_bert
            )
        
        # Freeze all other BERT parameters if specified
        if freeze_bert:
            for name, param in self.bert.named_parameters():
                if 'lora_' not in name:  # Don't freeze LoRA parameters
                    param.requires_grad = False
        
        # Classification head
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(pooled_output)

class TextDataset(Dataset):
    """Dataset for text classification tasks."""
    def __init__(self, texts: List[str], labels: List[int], tokenizer: AutoTokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'token_type_ids': encoding.get('token_type_ids', None),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 5,
    learning_rate: float = 1e-3,  # Higher learning rate for LoRA
    device: str = 'cuda'
) -> None:
    """
    Train the model using LoRA adaptations.
    Note the higher learning rate - we can use this because we're only
    training a small number of parameters.
    """
    model = model.to(device)
    
    # We only want to optimize the LoRA parameters
    optimizer = torch.optim.AdamW(
        [p for n, p in model.named_parameters() if 'lora_' in n or 'classifier' in n],
        lr=learning_rate
    )
    criterion = nn.CrossEntropyLoss()
    
    # Cosine learning rate scheduler with warmup
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2
    )
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Training phase
        model.train()
        train_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch.get('token_type_ids', None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch.get('token_type_ids', None)
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask, token_type_ids)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Accuracy: {accuracy:.2f}%")
        
        # Print the magnitudes of LoRA updates
        with torch.no_grad():
            lora_magnitudes = []
            for name, param in model.named_parameters():
                if 'lora_' in name:
                    lora_magnitudes.append(
                        (name, param.abs().mean().item())
                    )
            print("\nLoRA update magnitudes:")
            for name, mag in lora_magnitudes:
                print(f"{name}: {mag:.4f}")

def save_lora_weights(
    model: nn.Module,
    path: str,
    config: Optional[Dict] = None
) -> None:
    """
    Save only the LoRA weights and configuration.
    This is much more efficient than saving the entire model.
    """
    lora_state_dict = {
        name: param.cpu()
        for name, param in model.named_parameters()
        if 'lora_' in name or 'classifier' in name
    }
    
    if config is None:
        config = {
            'lora_rank': 8,
            'lora_alpha': 1.0,
            'num_classes': 2
        }
    
    torch.save({
        'lora_state_dict': lora_state_dict,
        'config': config
    }, path)

def load_lora_weights(
    base_model_name: str,
    path: str
) -> LoRABERTClassifier:
    """
    Load a model with saved LoRA weights.
    """
    checkpoint = torch.load(path)
    config = checkpoint['config']
    
    model = LoRABERTClassifier(
        model_name=base_model_name,
        num_classes=config['num_classes'],
        lora_rank=config['lora_rank'],
        lora_alpha=config['lora_alpha']
    )
    
    # Load only the LoRA weights
    model_state_dict = model.state_dict()
    for name, param in checkpoint['lora_state_dict'].items():
        if name in model_state_dict:
            model_state_dict[name].copy_(param)
    
    return model

# Example usage
def main():
    # Initialize tokenizer and model
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LoRABERTClassifier(
        model_name=model_name,
        num_classes=2,
        lora_rank=8,
        lora_alpha=1.0
    )
    
    # Example data (replace with your own dataset)
    texts = [
        "This is a positive review.",
        "This is a negative review.",
        # Add more examples...
    ]
    labels = [1, 0]  # Corresponding labels
    
    # Create datasets and dataloaders
    dataset = TextDataset(texts, labels, tokenizer)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    # Train the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_model(model, train_loader, val_loader, device=device)
    
    # Save only the LoRA weights
    save_lora_weights(model, 'lora_weights.pth')
    
    # Later, you can load just the LoRA weights onto a new model
    loaded_model = load_lora_weights(model_name, 'lora_weights.pth')

if __name__ == "__main__":
    main()
