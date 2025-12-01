"""
Data processing module for the Transformer model implementation.
Handles data loading, preprocessing, tokenization, and batching.
"""

import os
import math
import numpy as np
import torch
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence
from torchtext.data import Dataset, Example, Field, Iterator
from torchtext.vocab import Vocab, build_vocab_from_iterator
from torchtext.datasets import WMT14
from torchtext.data.utils import get_tokenizer
import sentencepiece as spm
from typing import List, Tuple, Dict, Iterator as IterType, Union, Optional, Callable

# Import from project modules
from config import Config
from utils import create_masks, create_padding_mask, create_subsequent_mask


class DataProcessor:
    """
    Handles data loading, preprocessing, tokenization, and batching.
    """
    def __init__(self, config: Config):
        """
        Initialize with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.data_config = config.get_data_params()
        self.model_config = config.get_model_params()
        self.device = config.device
        self.max_seq_length = self.model_config['max_seq_length']
        
        # Get language pair info
        self.source_lang = self.data_config['source_lang']
        self.target_lang = self.data_config['target_lang']
        
        # Set vocabulary size and tokenization method
        self.vocab_size = self.data_config['vocab_size']
        self.tokenization = self.data_config['tokenization']
        
        # Special tokens
        self.PAD_IDX = 0
        self.BOS_IDX = 1
        self.EOS_IDX = 2
        self.UNK_IDX = 3
        
        # Initialize vocabularies as None (to be built)
        self.src_vocab = None
        self.tgt_vocab = None
        
        # Initialize tokenizers
        self._init_tokenizers()
    
    def _init_tokenizers(self) -> None:
        """
        Initialize tokenizers based on configuration.
        """
        # Base tokenizer function (word-level tokenization)
        self.base_tokenizer = get_tokenizer('spacy', language=f'{self.source_lang}_core_web_sm')
        
        # Initialize BPE or WordPiece tokenizers if needed
        if self.tokenization == 'bpe':
            self._init_bpe_tokenizer()
        elif self.tokenization == 'wordpiece':
            self._init_wordpiece_tokenizer()
    
    def _init_bpe_tokenizer(self) -> None:
        """
        Initialize byte-pair encoding tokenizer.
        Will train the model if it doesn't exist or load a pre-trained model.
        """
        # Create directory for tokenizer models if it doesn't exist
        os.makedirs('tokenizers', exist_ok=True)
        
        # Define model path
        model_prefix = f'tokenizers/bpe_{self.source_lang}_{self.target_lang}'
        model_path = f'{model_prefix}.model'
        
        # Check if model exists, otherwise train it (when datasets are loaded)
        self.bpe_model_path = model_path
        self.bpe_model_prefix = model_prefix
        
        # We'll train or load the model later when data is available
    
    def _init_wordpiece_tokenizer(self) -> None:
        """
        Initialize WordPiece tokenizer.
        Will train the model if it doesn't exist or load a pre-trained model.
        """
        # Create directory for tokenizer models if it doesn't exist
        os.makedirs('tokenizers', exist_ok=True)
        
        # Define model path
        model_prefix = f'tokenizers/wp_{self.source_lang}_{self.target_lang}'
        model_path = f'{model_prefix}.model'
        
        # Check if model exists, otherwise train it (when datasets are loaded)
        self.wp_model_path = model_path
        self.wp_model_prefix = model_prefix
        
        # We'll train or load the model later when data is available
    
    def load_data(self, dataset_path: Optional[str] = None) -> Tuple[data.DataLoader, data.DataLoader, data.DataLoader]:
        """
        Load and prepare train/val/test data.
        
        Args:
            dataset_path: Path to dataset (optional, will use default if not provided)
            
        Returns:
            Tuple of (train_dataloader, val_dataloader, test_dataloader)
        """
        print(f"Loading {self.source_lang}-{self.target_lang} dataset...")
        
        # Create dataset paths
        if dataset_path is None:
            dataset_path = '.data'
        
        # Define dataset splits based on config
        train_split = self.data_config['train']
        valid_split = self.data_config['valid']
        test_split = self.data_config['test']
        
        # Create Fields for source and target
        src_field = Field(
            tokenize=self.tokenize,
            init_token='<bos>',
            eos_token='<eos>',
            pad_token='<pad>',
            unk_token='<unk>',
            lower=True,
            batch_first=True
        )
        
        tgt_field = Field(
            tokenize=self.tokenize,
            init_token='<bos>',
            eos_token='<eos>',
            pad_token='<pad>',
            unk_token='<unk>',
            lower=True,
            batch_first=True
        )
        
        # Specify fields for torchtext dataset
        fields = [(self.source_lang, src_field), (self.target_lang, tgt_field)]
        
        # Load datasets
        train_data, valid_data, test_data = WMT14.splits(
            exts=(f'.{self.source_lang}', f'.{self.target_lang}'),
            fields=fields,
            root=dataset_path,
            filter_pred=lambda x: len(vars(x)[self.source_lang]) <= self.max_seq_length and 
                                len(vars(x)[self.target_lang]) <= self.max_seq_length
        )
        
        print(f"Number of training examples: {len(train_data)}")
        print(f"Number of validation examples: {len(valid_data)}")
        print(f"Number of testing examples: {len(test_data)}")
        
        # Build vocabularies from training data
        if self.tokenization in ['bpe', 'wordpiece']:
            # For subword tokenization, we need to train the tokenizer first
            if self.tokenization == 'bpe':
                self._train_bpe_tokenizer(train_data)
            else:
                self._train_wordpiece_tokenizer(train_data)
            
            # Apply subword tokenization to the datasets
            train_data = self._apply_subword_tokenization(train_data)
            valid_data = self._apply_subword_tokenization(valid_data)
            test_data = self._apply_subword_tokenization(test_data)
        
        # Build vocabularies
        self.build_vocab(train_data, src_field, tgt_field)
        
        # Create bucketed iterators to efficiently batch sequences of similar lengths
        train_iterator, valid_iterator, test_iterator = self.batch_data(train_data, valid_data, test_data)
        
        # Convert iterators to PyTorch DataLoader format
        train_dataloader = self._convert_to_dataloader(train_iterator)
        valid_dataloader = self._convert_to_dataloader(valid_iterator)
        test_dataloader = self._convert_to_dataloader(test_iterator)
        
        return train_dataloader, valid_dataloader, test_dataloader
    
    def _train_bpe_tokenizer(self, train_data: Dataset) -> None:
        """
        Train byte-pair encoding tokenizer on training data.
        
        Args:
            train_data: Training dataset
        """
        # Check if model already exists
        if os.path.exists(self.bpe_model_path):
            print(f"Loading existing BPE model from {self.bpe_model_path}")
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(self.bpe_model_path)
            return
        
        # Create corpus file for training
        corpus_file = f'{self.bpe_model_prefix}.corpus'
        with open(corpus_file, 'w', encoding='utf-8') as f:
            # Write source sentences
            for example in train_data.examples:
                f.write(' '.join(vars(example)[self.source_lang]) + '\n')
            
            # Write target sentences
            for example in train_data.examples:
                f.write(' '.join(vars(example)[self.target_lang]) + '\n')
        
        # Train SentencePiece model
        print(f"Training BPE tokenizer with vocabulary size {self.vocab_size}")
        spm.SentencePieceTrainer.train(
            f'--input={corpus_file} '
            f'--model_prefix={self.bpe_model_prefix} '
            f'--vocab_size={self.vocab_size} '
            f'--character_coverage=0.9995 '
            f'--model_type=bpe '
            f'--pad_id=0 --bos_id=1 --eos_id=2 --unk_id=3 '
            f'--user_defined_symbols=<pad>,<bos>,<eos>,<unk>'
        )
        
        # Load trained model
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.bpe_model_path)
        
        # Remove corpus file
        os.remove(corpus_file)
    
    def _train_wordpiece_tokenizer(self, train_data: Dataset) -> None:
        """
        Train WordPiece tokenizer on training data.
        
        Args:
            train_data: Training dataset
        """
        # Check if model already exists
        if os.path.exists(self.wp_model_path):
            print(f"Loading existing WordPiece model from {self.wp_model_path}")
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(self.wp_model_path)
            return
        
        # Create corpus file for training
        corpus_file = f'{self.wp_model_prefix}.corpus'
        with open(corpus_file, 'w', encoding='utf-8') as f:
            # Write source sentences
            for example in train_data.examples:
                f.write(' '.join(vars(example)[self.source_lang]) + '\n')
            
            # Write target sentences
            for example in train_data.examples:
                f.write(' '.join(vars(example)[self.target_lang]) + '\n')
        
        # Train SentencePiece model with WordPiece
        print(f"Training WordPiece tokenizer with vocabulary size {self.vocab_size}")
        spm.SentencePieceTrainer.train(
            f'--input={corpus_file} '
            f'--model_prefix={self.wp_model_prefix} '
            f'--vocab_size={self.vocab_size} '
            f'--character_coverage=0.9995 '
            f'--model_type=word '
            f'--pad_id=0 --bos_id=1 --eos_id=2 --unk_id=3 '
            f'--user_defined_symbols=<pad>,<bos>,<eos>,<unk>'
        )
        
        # Load trained model
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.wp_model_path)
        
        # Remove corpus file
        os.remove(corpus_file)
    
    def _apply_subword_tokenization(self, dataset: Dataset) -> Dataset:
        """
        Apply subword tokenization to a dataset.
        
        Args:
            dataset: Dataset to tokenize
            
        Returns:
            Tokenized dataset
        """
        # Create a new dataset with subword-tokenized examples
        examples = []
        for example in dataset.examples:
            src_text = ' '.join(vars(example)[self.source_lang])
            tgt_text = ' '.join(vars(example)[self.target_lang])
            
            # Apply subword tokenization
            src_tokens = self.sp.encode(src_text, out_type=str)
            tgt_tokens = self.sp.encode(tgt_text, out_type=str)
            
            # Create a new example with tokenized text
            new_example = Example()
            setattr(new_example, self.source_lang, src_tokens)
            setattr(new_example, self.target_lang, tgt_tokens)
            examples.append(new_example)
        
        # Create new dataset with tokenized examples
        return Dataset(examples, dataset.fields)
    
    def build_vocab(self, train_data: Dataset, src_field: Field, tgt_field: Field) -> Tuple[Vocab, Vocab]:
        """
        Build source and target vocabularies.
        
        Args:
            train_data: Training dataset
            src_field: Source field
            tgt_field: Target field
            
        Returns:
            Tuple of (source vocabulary, target vocabulary)
        """
        if self.tokenization in ['bpe', 'wordpiece']:
            # For subword tokenization, use the vocabulary from SentencePiece
            sp_vocab = {self.sp.id_to_piece(i): i for i in range(self.sp.get_piece_size())}
            src_field.vocab = Vocab(sp_vocab, specials=[])
            tgt_field.vocab = Vocab(sp_vocab, specials=[])
        else:
            # For word-level tokenization, build vocabulary from training data
            src_field.build_vocab(train_data, max_size=self.vocab_size)
            tgt_field.build_vocab(train_data, max_size=self.vocab_size)
        
        # Store vocabularies
        self.src_vocab = src_field.vocab
        self.tgt_vocab = tgt_field.vocab
        
        print(f"Source vocabulary size: {len(self.src_vocab)}")
        print(f"Target vocabulary size: {len(self.tgt_vocab)}")
        
        return self.src_vocab, self.tgt_vocab
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Use base tokenizer for initial tokenization
        tokens = self.base_tokenizer(text)
        return tokens
    
    def apply_bpe(self, tokens: List[str]) -> List[str]:
        """
        Apply byte-pair encoding to tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of BPE tokens
        """
        # Join tokens and apply BPE
        text = ' '.join(tokens)
        bpe_tokens = self.sp.encode(text, out_type=str)
        return bpe_tokens
    
    def batch_data(self, train_data: Dataset, valid_data: Dataset, test_data: Dataset) -> Tuple[Iterator, Iterator, Iterator]:
        """
        Create batches of similar lengths.
        
        Args:
            train_data: Training dataset
            valid_data: Validation dataset
            test_data: Test dataset
            
        Returns:
            Tuple of (train iterator, validation iterator, test iterator)
        """
        # Calculate batch size based on target tokens per batch
        # We'll do dynamic batching in the bucket iterator
        batch_size = self.config.batch_tokens // self.max_seq_length
        batch_size = max(1, batch_size)  # Ensure at least 1
        
        # Create BucketIterator for batching similar-length sequences
        train_iterator, valid_iterator, test_iterator = Iterator.splits(
            (train_data, valid_data, test_data),
            batch_size=batch_size,
            sort_key=lambda x: len(getattr(x, self.source_lang)),
            sort_within_batch=True,
            device=self.device
        )
        
        return train_iterator, valid_iterator, test_iterator
    
    def _convert_to_dataloader(self, iterator: Iterator) -> data.DataLoader:
        """
        Convert torchtext iterator to PyTorch DataLoader.
        
        Args:
            iterator: torchtext iterator
            
        Returns:
            PyTorch DataLoader
        """
        # Create a dataset that yields batches from the iterator
        dataset = _IteratorDataset(iterator, self.source_lang, self.target_lang)
        
        # Create a DataLoader with the dataset
        return data.DataLoader(
            dataset,
            batch_size=None,  # Batching is already done by the iterator
            collate_fn=None   # No need for collation
        )
    
    def create_masks(self, src: torch.Tensor, tgt: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Create attention masks for the transformer model.
        
        Args:
            src: Source tensor of shape (batch_size, src_len)
            tgt: Target tensor of shape (batch_size, tgt_len) (optional, for training)
            
        Returns:
            Tuple of (source mask, target mask) where target mask is None if tgt is None
        """
        return create_masks(src, tgt, self.PAD_IDX)
    
    def encode_sentence(self, sentence: str, is_source: bool = True) -> torch.Tensor:
        """
        Encode a sentence to tensor with vocabulary indices.
        
        Args:
            sentence: Sentence to encode
            is_source: Whether this is a source sentence (or target)
            
        Returns:
            Tensor with vocabulary indices
        """
        # Tokenize the sentence
        tokens = self.tokenize(sentence)
        
        # Apply subword tokenization if needed
        if self.tokenization in ['bpe', 'wordpiece']:
            tokens = self.sp.encode(' '.join(tokens), out_type=str)
        
        # Get vocabulary (source or target)
        vocab = self.src_vocab if is_source else self.tgt_vocab
        
        # Convert tokens to indices
        indices = [vocab.stoi.get(token, self.UNK_IDX) for token in tokens]
        
        # Add BOS and EOS tokens
        indices = [self.BOS_IDX] + indices + [self.EOS_IDX]
        
        # Convert to tensor
        return torch.tensor(indices, dtype=torch.long).unsqueeze(0)
    
    def decode_sentence(self, indices: torch.Tensor, is_source: bool = False) -> str:
        """
        Decode indices to sentence.
        
        Args:
            indices: Tensor with vocabulary indices
            is_source: Whether this is a source sentence (or target)
            
        Returns:
            Decoded sentence
        """
        # Get vocabulary (source or target)
        vocab = self.src_vocab if is_source else self.tgt_vocab
        
        # Convert indices to tokens, skipping special tokens
        tokens = []
        for idx in indices:
            if idx == self.EOS_IDX:
                break
            if idx != self.BOS_IDX and idx != self.PAD_IDX:
                tokens.append(vocab.itos[idx])
        
        # For subword tokenization, join with spaces and then remove separators
        if self.tokenization in ['bpe', 'wordpiece']:
            return self.sp.decode(tokens)
        else:
            # For word-level tokenization, join with spaces
            return ' '.join(tokens)


class _IteratorDataset(data.Dataset):
    """
    Dataset adapter for torchtext iterator to PyTorch DataLoader.
    """
    def __init__(self, iterator: Iterator, src_field: str, tgt_field: str):
        """
        Initialize with iterator and field names.
        
        Args:
            iterator: torchtext iterator
            src_field: Source field name
            tgt_field: Target field name
        """
        self.iterator = iterator
        self.src_field = src_field
        self.tgt_field = tgt_field
        self._iterator = iter(iterator)
        self.length = len(iterator)
    
    def __len__(self) -> int:
        """Get dataset length."""
        return self.length
    
    def __iter__(self) -> IterType:
        """Reset and return iterator."""
        self._iterator = iter(self.iterator)
        return self
    
    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get next batch."""
        batch = next(self._iterator)
        src = getattr(batch, self.src_field)
        tgt = getattr(batch, self.tgt_field)
        return src, tgt
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item is not meaningful for iterator, but needed for DataLoader."""
        raise NotImplementedError("This dataset does not support random access")
