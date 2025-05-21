"""
Training module for the Transformer model implementation.
Handles the training and validation loops for the Transformer model.
"""

import os
import time
import math
from typing import Optional, Tuple, Dict, List, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Import from project modules
from config import Config
from model import TransformerModel
from data_processing import DataProcessor
from utils import (
    label_smoothed_nll_loss, 
    get_lr_scheduler, 
    save_checkpoint, 
    load_checkpoint,
    create_masks
)


class Trainer:
    """
    Handles the training and validation loops for the Transformer model.
    """
    def __init__(
        self, 
        config: Config, 
        model: TransformerModel, 
        data_processor: DataProcessor
    ):
        """
        Initialize trainer.
        
        Args:
            config: Configuration object
            model: TransformerModel instance
            data_processor: DataProcessor instance
        """
        self.config = config
        self.model = model
        self.data_processor = data_processor
        
        # Get training parameters
        training_params = config.get_training_params()
        self.device = training_params['device']
        self.warmup_steps = training_params['warmup_steps']
        self.label_smoothing = training_params['label_smoothing']
        self.total_steps = training_params['total_steps']
        self.checkpoint_interval = training_params['checkpoint_interval']
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer with betas and epsilon as specified in the paper
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=0.0,  # Will be set by scheduler
            betas=(training_params['beta1'], training_params['beta2']),
            eps=training_params['epsilon']
        )
        
        # Initialize learning rate scheduler
        self.lr_scheduler = get_lr_scheduler(
            self.optimizer, 
            config.get_model_params()['d_model'], 
            self.warmup_steps
        )
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(log_dir="logs")
        
        # Track training statistics
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
    def train(
        self, 
        train_data: DataLoader, 
        val_data: DataLoader, 
        epochs: Optional[int] = None, 
        checkpoint_dir: str = "checkpoints"
    ) -> None:
        """
        Main training loop.
        
        Args:
            train_data: Training data loader
            val_data: Validation data loader
            epochs: Number of epochs to train (if None, will use total_steps)
            checkpoint_dir: Directory to save checkpoints
        """
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Start time for checkpoint saving
        last_checkpoint_time = time.time()
        
        print(f"Starting training on device: {self.device}")
        print(f"Model size: {self.config.model_size}")
        
        # Main training loop
        while True:
            self.epoch += 1
            print(f"\nEpoch {self.epoch}")
            
            # Train for one epoch
            train_loss = self.train_epoch(train_data)
            
            # Evaluate on validation data
            val_loss = self.validate(val_data)
            
            # Log losses
            self.writer.add_scalar('Loss/train', train_loss, self.epoch)
            self.writer.add_scalar('Loss/val', val_loss, self.epoch)
            
            print(f"Epoch {self.epoch}: Train loss = {train_loss:.4f}, Val loss = {val_loss:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(os.path.join(checkpoint_dir, "best_model.pt"))
                print(f"New best model saved with validation loss: {val_loss:.4f}")
            
            # Save checkpoint periodically
            current_time = time.time()
            if current_time - last_checkpoint_time > self.checkpoint_interval * 60:  # Convert to seconds
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{self.epoch}_step{self.step}.pt")
                self.save_checkpoint(checkpoint_path)
                last_checkpoint_time = current_time
            
            # Check stopping conditions
            if self.step >= self.total_steps:
                print(f"Reached {self.total_steps} steps. Training complete.")
                # Save final model
                self.save_checkpoint(os.path.join(checkpoint_dir, "final_model.pt"))
                break
            
            if epochs is not None and self.epoch >= epochs:
                print(f"Reached {epochs} epochs. Training complete.")
                # Save final model
                self.save_checkpoint(os.path.join(checkpoint_dir, "final_model.pt"))
                break
    
    def train_epoch(self, train_data: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_data: Training data loader
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0
        total_tokens = 0
        start_time = time.time()
        
        # Create progress bar
        pbar = tqdm(train_data, desc=f"Training epoch {self.epoch}")
        
        for i, (src, tgt) in enumerate(pbar):
            # Move data to device
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            
            # Create masks
            src_mask, tgt_mask = self.data_processor.create_masks(src, tgt)
            
            # Prepare target for loss calculation (shift by 1)
            # Input: <bos> w1 w2 w3
            # Target: w1 w2 w3 <eos>
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # Create mask for target input
            _, tgt_input_mask = self.data_processor.create_masks(src, tgt_input)
            
            # Forward pass
            logits = self.model(src, tgt_input, src_mask, tgt_input_mask)
            
            # Flatten logits and targets for loss calculation
            logits = logits.contiguous().view(-1, logits.size(-1))
            tgt_output = tgt_output.contiguous().view(-1)
            
            # Calculate loss with label smoothing
            loss, nll_loss = label_smoothed_nll_loss(
                logits,
                tgt_output,
                self.label_smoothing,
                ignore_index=self.data_processor.PAD_IDX
            )
            
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients as mentioned in paper (not explicitly stated value, using common default)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Adjust learning rate according to schedule
            lr = self.lr_scheduler(self.step)
            
            # Update parameters
            self.optimizer.step()
            
            # Update statistics
            self.step += 1
            total_loss += nll_loss.item() * tgt_output.ne(self.data_processor.PAD_IDX).sum().item()
            total_tokens += tgt_output.ne(self.data_processor.PAD_IDX).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{lr:.7f}",
                'step': self.step
            })
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/train_step', loss.item(), self.step)
            self.writer.add_scalar('Learning rate', lr, self.step)
            
            # Check if total steps reached
            if self.step >= self.total_steps:
                break
        
        # Calculate average loss
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        
        # Calculate training time
        elapsed_time = time.time() - start_time
        tokens_per_sec = total_tokens / elapsed_time if elapsed_time > 0 else 0
        
        print(f"Epoch {self.epoch} completed in {elapsed_time:.2f} seconds")
        print(f"Training throughput: {tokens_per_sec:.2f} tokens/sec")
        
        return avg_loss
    
    def validate(self, val_data: DataLoader) -> float:
        """
        Validate model.
        
        Args:
            val_data: Validation data loader
            
        Returns:
            Validation loss
        """
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        # Create progress bar
        pbar = tqdm(val_data, desc=f"Validating epoch {self.epoch}")
        
        with torch.no_grad():
            for src, tgt in pbar:
                # Move data to device
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                
                # Create masks
                src_mask, tgt_mask = self.data_processor.create_masks(src, tgt)
                
                # Prepare target for loss calculation (shift by 1)
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                # Create mask for target input
                _, tgt_input_mask = self.data_processor.create_masks(src, tgt_input)
                
                # Forward pass
                logits = self.model(src, tgt_input, src_mask, tgt_input_mask)
                
                # Flatten logits and targets for loss calculation
                logits = logits.contiguous().view(-1, logits.size(-1))
                tgt_output = tgt_output.contiguous().view(-1)
                
                # Calculate loss with label smoothing
                _, nll_loss = label_smoothed_nll_loss(
                    logits,
                    tgt_output,
                    self.label_smoothing,
                    ignore_index=self.data_processor.PAD_IDX
                )
                
                # Update statistics
                total_loss += nll_loss.item() * tgt_output.ne(self.data_processor.PAD_IDX).sum().item()
                total_tokens += tgt_output.ne(self.data_processor.PAD_IDX).sum().item()
        
        # Calculate average loss
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        
        return avg_loss
    
    def save_checkpoint(self, path: str) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        save_checkpoint(
            self.model,
            self.optimizer,
            self.epoch,
            self.step,
            self.best_val_loss,
            path
        )
    
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            path: Path to load checkpoint from
            
        Returns:
            Dictionary with checkpoint metadata
        """
        metadata = load_checkpoint(path, self.model, self.optimizer)
        
        # Update trainer state
        self.epoch = metadata.get('epoch', 0)
        self.step = metadata.get('step', 0)
        self.best_val_loss = metadata.get('loss', float('inf'))
        
        print(f"Loaded checkpoint from {path}")
        print(f"Epoch: {self.epoch}, Step: {self.step}, Best val loss: {self.best_val_loss:.4f}")
        
        return metadata
    
    def adjust_learning_rate(self, step: int) -> float:
        """
        Adjust learning rate according to schedule.
        
        Args:
            step: Current step number
            
        Returns:
            New learning rate
        """
        return self.lr_scheduler(step)


if __name__ == "__main__":
    # Simple test for the trainer
    print("Testing Trainer class...")
    
    # Load configuration
    config = Config(model_size='base')
    
    # Create dummy data processor
    data_processor = DataProcessor(config)
    
    # Create dummy model
    model = TransformerModel(
        config,
        src_vocab_size=1000,  # Dummy value
        tgt_vocab_size=1000   # Dummy value
    )
    
    # Create trainer
    trainer = Trainer(config, model, data_processor)
    
    print("Trainer initialized successfully!")
    print(f"Model will be trained for {trainer.total_steps} steps with {trainer.warmup_steps} warmup steps")
    print(f"Label smoothing: {trainer.label_smoothing}")
    print(f"Device: {trainer.device}")
