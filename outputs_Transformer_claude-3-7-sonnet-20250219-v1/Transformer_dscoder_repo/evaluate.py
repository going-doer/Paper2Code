"""
Evaluation module for the Transformer model implementation.
Handles evaluation and inference for the trained model.
"""

import os
import time
import math
from typing import Optional, Tuple, Dict, List, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sacrebleu
from tqdm import tqdm

# Import from project modules
from config import Config
from model import TransformerModel
from data_processing import DataProcessor
from utils import average_checkpoints, create_masks


class Evaluator:
    """
    Handles evaluation and inference for the trained Transformer model.
    """
    def __init__(
        self, 
        config: Config, 
        model: TransformerModel, 
        data_processor: DataProcessor
    ):
        """
        Initialize evaluator.
        
        Args:
            config: Configuration object
            model: TransformerModel instance
            data_processor: DataProcessor instance
        """
        self.config = config
        self.model = model
        self.data_processor = data_processor
        
        # Get inference parameters
        inference_params = config.get_inference_params()
        self.device = inference_params['device']
        self.beam_size = inference_params['beam_size']
        self.length_penalty = inference_params['length_penalty']
        self.max_length_factor = inference_params['max_length_factor']
        
        # Move model to device
        self.model.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
    def evaluate(self, test_data: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            test_data: Test data loader
            
        Returns:
            Dictionary containing evaluation metrics (e.g., BLEU score)
        """
        self.model.eval()
        
        # Lists to store references and hypotheses
        references = []
        hypotheses = []
        
        print("Evaluating model...")
        
        # Create progress bar
        pbar = tqdm(test_data, desc="Evaluating")
        
        with torch.no_grad():
            for src, tgt in pbar:
                # Move data to device
                src = src.to(self.device)
                
                # Get batch size and max possible length
                batch_size = src.size(0)
                max_len = min(src.size(1) + self.max_length_factor, 
                             self.config.model_config['max_seq_length'])
                
                # Generate translations using beam search
                generated = self.model.beam_search(
                    src, 
                    max_len, 
                    start_symbol=self.data_processor.BOS_IDX, 
                    end_symbol=self.data_processor.EOS_IDX
                )
                
                # Convert tensors to sentences
                for i in range(batch_size):
                    # Get reference sentence (target)
                    ref_sentence = self.data_processor.decode_sentence(tgt[i], is_source=False)
                    references.append(ref_sentence)
                    
                    # Get hypothesis sentence (generated)
                    hyp_sentence = self.data_processor.decode_sentence(generated[i], is_source=False)
                    hypotheses.append(hyp_sentence)
        
        # Calculate BLEU score
        bleu_score = self.compute_bleu(references, hypotheses)
        
        print(f"BLEU score: {bleu_score}")
        
        # Return metrics
        return {
            'bleu': bleu_score,
            'num_samples': len(references)
        }
    
    def translate_sentence(self, sentence: str) -> str:
        """
        Translate a single sentence.
        
        Args:
            sentence: Input sentence in source language
            
        Returns:
            Translated sentence in target language
        """
        self.model.eval()
        
        # Encode the sentence
        src_tensor = self.data_processor.encode_sentence(sentence, is_source=True).to(self.device)
        
        # Calculate maximum output length
        max_len = min(src_tensor.size(1) + self.max_length_factor, 
                     self.config.model_config['max_seq_length'])
        
        # Generate translation
        with torch.no_grad():
            generated = self.model.beam_search(
                src_tensor, 
                max_len, 
                start_symbol=self.data_processor.BOS_IDX, 
                end_symbol=self.data_processor.EOS_IDX
            )
        
        # Decode the generated tensor to text
        translation = self.data_processor.decode_sentence(generated[0], is_source=False)
        
        return translation
    
    def compute_bleu(self, references: List[str], hypotheses: List[str]) -> float:
        """
        Compute BLEU score using sacrebleu.
        
        Args:
            references: List of reference sentences
            hypotheses: List of hypothesis sentences
            
        Returns:
            BLEU score
        """
        # Convert single references to list of lists format required by sacrebleu
        references_list = [[ref] for ref in references]
        
        # Calculate corpus BLEU score
        bleu = sacrebleu.corpus_bleu(hypotheses, references_list)
        
        # Return the score as a float
        return bleu.score
    
    def average_checkpoints(self, paths: List[str]) -> None:
        """
        Average model weights from multiple checkpoints as described in the paper.
        
        Args:
            paths: List of paths to checkpoints
        """
        if not paths:
            print("No checkpoint paths provided for averaging.")
            return
        
        print(f"Averaging {len(paths)} checkpoints...")
        average_checkpoints(paths, self.model)
        print("Checkpoint averaging complete.")
    
    def find_latest_checkpoints(
        self, 
        checkpoint_dir: str, 
        num_checkpoints: int
    ) -> List[str]:
        """
        Find the latest checkpoints in a directory.
        
        Args:
            checkpoint_dir: Directory containing checkpoints
            num_checkpoints: Number of latest checkpoints to find
            
        Returns:
            List of paths to latest checkpoints
        """
        # List all checkpoint files
        checkpoint_files = [
            os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir)
            if f.startswith('checkpoint_') and f.endswith('.pt')
        ]
        
        # Sort by modification time (newest first)
        checkpoint_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        # Return the specified number of checkpoints
        return checkpoint_files[:num_checkpoints]
    
    def generate_translations(
        self, 
        src_sentences: List[str], 
        output_file: Optional[str] = None
    ) -> List[str]:
        """
        Generate translations for a list of source sentences.
        
        Args:
            src_sentences: List of source language sentences
            output_file: Path to write translations to (optional)
            
        Returns:
            List of translated sentences
        """
        self.model.eval()
        translations = []
        
        # Create progress bar
        pbar = tqdm(src_sentences, desc="Generating translations")
        
        for sentence in pbar:
            translation = self.translate_sentence(sentence)
            translations.append(translation)
        
        # Write translations to file if specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                for trans in translations:
                    f.write(trans + '\n')
            print(f"Translations written to {output_file}")
        
        return translations
    
    def evaluate_from_checkpoint(self, checkpoint_path: str, test_data: DataLoader) -> Dict[str, float]:
        """
        Load a checkpoint and evaluate the model.
        
        Args:
            checkpoint_path: Path to checkpoint
            test_data: Test data loader
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        
        # Load model state
        self.model.load_state_dict(checkpoint['model'])
        
        # Move model to device
        self.model.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        
        # Evaluate model
        return self.evaluate(test_data)
    
    def evaluate_averaged_model(
        self, 
        checkpoint_dir: str, 
        test_data: DataLoader, 
        num_checkpoints: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Average checkpoints and evaluate the resulting model.
        
        Args:
            checkpoint_dir: Directory containing checkpoints
            test_data: Test data loader
            num_checkpoints: Number of checkpoints to average (if None, use config value)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Get number of checkpoints to average from config if not specified
        if num_checkpoints is None:
            num_checkpoints = self.config.average_checkpoints
        
        # Find the latest checkpoints
        checkpoint_paths = self.find_latest_checkpoints(checkpoint_dir, num_checkpoints)
        
        if not checkpoint_paths:
            raise ValueError(f"No checkpoints found in {checkpoint_dir}")
        
        # Average the checkpoints
        self.average_checkpoints(checkpoint_paths)
        
        # Evaluate the averaged model
        return self.evaluate(test_data)
    
    def compute_perplexity(self, data_loader: DataLoader) -> float:
        """
        Compute perplexity on a dataset.
        
        Args:
            data_loader: Data loader for evaluation
            
        Returns:
            Perplexity score
        """
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for src, tgt in tqdm(data_loader, desc="Computing perplexity"):
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
                
                # Calculate loss
                loss = F.cross_entropy(
                    logits, 
                    tgt_output, 
                    ignore_index=self.data_processor.PAD_IDX,
                    reduction='sum'
                )
                
                # Update statistics
                total_loss += loss.item()
                total_tokens += tgt_output.ne(self.data_processor.PAD_IDX).sum().item()
        
        # Calculate perplexity
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = math.exp(avg_loss)
        
        print(f"Perplexity: {perplexity:.2f}")
        
        return perplexity


if __name__ == "__main__":
    # Simple test for the evaluator
    print("Testing Evaluator class...")
    
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
    
    # Create evaluator
    evaluator = Evaluator(config, model, data_processor)
    
    print("Evaluator initialized successfully!")
    print(f"Beam size: {evaluator.beam_size}")
    print(f"Length penalty: {evaluator.length_penalty}")
    print(f"Device: {evaluator.device}")
