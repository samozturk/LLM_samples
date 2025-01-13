from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Optional

class CustomBERTClassifier(nn.Module):
    def __init__(
        self, 
        model_name: str = "bert-base-uncased",
        num_classes: int = 2,
        dropout_rate: float = 0.3,
        freeze_bert: bool = True
    ):
        """
        Initialize a custom BERT classifier with additional layers.
        
        Args:
            model_name: Name of the pretrained model from HuggingFace
            num_classes: Number of output classes
            dropout_rate: Dropout probability for regularization
            freeze_bert: Whether to freeze the BERT parameters
        """
        super().__init__()
        
        # Load pretrained BERT model
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Freeze BERT parameters if specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Get the output dimension of BERT
        bert_output_dim = self.bert.config.hidden_size
        
        # Create custom layers for fine-tuning
        self.classifier = nn.Sequential(
            # First layer with batch normalization and dropout
            nn.Linear(bert_output_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Second layer with reduced dimensions
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Output layer
            nn.Linear(256, num_classes)
        )
        
        # Initialize the weights of our custom layers
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize the weights of the custom layers using Xavier initialization."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Tokenized input sequences
            attention_mask: Attention mask for padding
            token_type_ids: Optional segment ids
        
        Returns:
            Logits for each class
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Get the [CLS] token representation (first token)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Pass through our custom classifier
        return self.classifier(pooled_output)

class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: AutoTokenizer):
        """
        Custom dataset for text classification.
        
        Args:
            texts: List of input texts
            labels: List of corresponding labels
            tokenizer: HuggingFace tokenizer
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Tokenize text
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
    learning_rate: float = 2e-5,
    device: str = 'cuda'
) -> None:
    """
    Train the model with the given parameters.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        device: Device to train on ('cuda' or 'cpu')
    """
    model = model.to(device)
    
    # Initialize optimizer and loss function
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate
    )
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=2, verbose=True
    )
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Training phase
        model.train()
        train_loss = 0
        for batch in train_loader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch.get('token_type_ids', None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
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
        
        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)

# Example usage
def main():
    # Initialize tokenizer and model
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = CustomBERTClassifier(model_name=model_name, num_classes=2)
    
    # Example data (replace with your own dataset)
    texts = [
        "This is a positive review.",
        "This is a negative review.",
        # Add more examples...
    ]
    labels = [1, 0]  # Corresponding labels
    
    # Create datasets
    dataset = TextDataset(texts, labels, tokenizer)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16
    )
    
    # Train the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_model(model, train_loader, val_loader, device=device)
    
    # Save the fine-tuned model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'model_name': model_name,
            'num_classes': 2,
            'dropout_rate': 0.3
        }
    }, 'fine_tuned_bert.pth')

if __name__ == "__main__":
    main()
