"""
Main module for the Transformer model implementation.
Entry point for running training and evaluation of the model.
"""

import os
import argparse
import torch
import time
from typing import Optional, List, Tuple

# Import from project modules
from config import Config
from model import TransformerModel
from data_processing import DataProcessor
from train import Trainer
from evaluate import Evaluator
from utils import average_checkpoints


def train_model(
    model_size: str = 'base',
    language_pair: str = 'en_de',
    config_path: Optional[str] = None,
    checkpoint_dir: str = "checkpoints",
    resume_checkpoint: Optional[str] = None,
    epochs: Optional[int] = None
) -> None:
    """
    Train the transformer model.
    
    Args:
        model_size: Size of the model ('base' or 'big')
        language_pair: Language pair to train on ('en_de' or 'en_fr')
        config_path: Path to configuration file
        checkpoint_dir: Directory to save checkpoints
        resume_checkpoint: Path to checkpoint to resume training from
        epochs: Number of epochs to train (if None, will use steps from config)
    """
    # Load configuration
    config = Config(model_size=model_size, config_path=config_path)
    
    # Set language pair
    config.set_language_pair(language_pair)
    
    # Create directory for checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize data processor
    print("Initializing data processor...")
    data_processor = DataProcessor(config)
    
    # Load data
    print(f"Loading {language_pair} dataset...")
    train_data, val_data, test_data = data_processor.load_data()
    
    # Get vocabulary sizes
    src_vocab_size = len(data_processor.src_vocab)
    tgt_vocab_size = len(data_processor.tgt_vocab)
    
    # Initialize model
    print(f"Initializing {model_size} transformer model...")
    model = TransformerModel(config, src_vocab_size, tgt_vocab_size)
    
    # Count model parameters
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model has {param_count:,} parameters")
    
    # Initialize trainer
    trainer = Trainer(config, model, data_processor)
    
    # Resume from checkpoint if specified
    if resume_checkpoint:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        trainer.load_checkpoint(resume_checkpoint)
    
    # Train model
    print("Starting training...")
    trainer.train(train_data, val_data, epochs=epochs, checkpoint_dir=checkpoint_dir)
    
    print("Training complete!")


def evaluate_model(
    model_path: str,
    model_size: str = 'base',
    language_pair: str = 'en_de',
    config_path: Optional[str] = None,
    is_averaged: bool = False,
    checkpoint_dir: Optional[str] = None,
    num_checkpoints: Optional[int] = None,
    output_file: Optional[str] = None
) -> float:
    """
    Evaluate the trained model.
    
    Args:
        model_path: Path to trained model
        model_size: Size of the model ('base' or 'big')
        language_pair: Language pair to evaluate on ('en_de' or 'en_fr')
        config_path: Path to configuration file
        is_averaged: Whether to average checkpoints
        checkpoint_dir: Directory containing checkpoints (for averaging)
        num_checkpoints: Number of checkpoints to average
        output_file: Path to write translations to
        
    Returns:
        BLEU score
    """
    # Load configuration
    config = Config(model_size=model_size, config_path=config_path)
    
    # Set language pair
    config.set_language_pair(language_pair)
    
    # Initialize data processor
    print("Initializing data processor...")
    data_processor = DataProcessor(config)
    
    # Load data
    print(f"Loading {language_pair} test dataset...")
    _, _, test_data = data_processor.load_data()
    
    # Get vocabulary sizes
    src_vocab_size = len(data_processor.src_vocab)
    tgt_vocab_size = len(data_processor.tgt_vocab)
    
    # Initialize model
    print(f"Initializing {model_size} transformer model...")
    model = TransformerModel(config, src_vocab_size, tgt_vocab_size)
    
    # Initialize evaluator
    evaluator = Evaluator(config, model, data_processor)
    
    if is_averaged and checkpoint_dir:
        # Average checkpoints
        if num_checkpoints is None:
            # Use default from config
            num_checkpoints = config.average_checkpoints
        
        print(f"Averaging {num_checkpoints} checkpoints from {checkpoint_dir}...")
        checkpoint_paths = evaluator.find_latest_checkpoints(checkpoint_dir, num_checkpoints)
        evaluator.average_checkpoints(checkpoint_paths)
    else:
        # Load single model
        print(f"Loading model from {model_path}...")
        evaluator.evaluate_from_checkpoint(model_path, test_data)
    
    # Evaluate model
    print("Evaluating model...")
    eval_results = evaluator.evaluate(test_data)
    
    # Print results
    bleu_score = eval_results['bleu']
    print(f"BLEU score: {bleu_score:.2f}")
    
    # Generate translations for test set and save to file if specified
    if output_file:
        print(f"Generating translations and saving to {output_file}...")
        # Extract source sentences from test data
        src_sentences = []
        for batch in test_data:
            src, _ = batch
            for i in range(src.size(0)):
                src_sentence = data_processor.decode_sentence(src[i], is_source=True)
                src_sentences.append(src_sentence)
        
        # Generate translations
        evaluator.generate_translations(src_sentences, output_file)
    
    return bleu_score


def translate(
    model_path: str,
    sentence: str,
    model_size: str = 'base',
    language_pair: str = 'en_de',
    config_path: Optional[str] = None
) -> str:
    """
    Translate a single sentence.
    
    Args:
        model_path: Path to trained model
        sentence: Sentence to translate
        model_size: Size of the model ('base' or 'big')
        language_pair: Language pair to translate ('en_de' or 'en_fr')
        config_path: Path to configuration file
        
    Returns:
        Translated sentence
    """
    # Load configuration
    config = Config(model_size=model_size, config_path=config_path)
    
    # Set language pair
    config.set_language_pair(language_pair)
    
    # Initialize data processor
    print("Initializing data processor...")
    data_processor = DataProcessor(config)
    
    # Load vocabularies
    print("Building vocabularies...")
    # Need to load some data to build vocabularies
    data_processor.load_data()
    
    # Get vocabulary sizes
    src_vocab_size = len(data_processor.src_vocab)
    tgt_vocab_size = len(data_processor.tgt_vocab)
    
    # Initialize model
    print(f"Initializing {model_size} transformer model...")
    model = TransformerModel(config, src_vocab_size, tgt_vocab_size)
    
    # Load model
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    
    # Initialize evaluator
    evaluator = Evaluator(config, model, data_processor)
    
    # Translate sentence
    print("Translating sentence...")
    translation = evaluator.translate_sentence(sentence)
    
    return translation


def average_model_checkpoints(
    checkpoint_dir: str,
    output_path: str,
    model_size: str = 'base',
    language_pair: str = 'en_de',
    config_path: Optional[str] = None,
    num_checkpoints: Optional[int] = None
) -> None:
    """
    Average multiple model checkpoints and save the result.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        output_path: Path to save the averaged model
        model_size: Size of the model ('base' or 'big')
        language_pair: Language pair ('en_de' or 'en_fr')
        config_path: Path to configuration file
        num_checkpoints: Number of checkpoints to average
    """
    # Load configuration
    config = Config(model_size=model_size, config_path=config_path)
    
    # Set language pair
    config.set_language_pair(language_pair)
    
    # Initialize data processor (needed for vocabulary sizes)
    print("Initializing data processor...")
    data_processor = DataProcessor(config)
    
    # Load vocabularies
    print("Building vocabularies...")
    data_processor.load_data()
    
    # Get vocabulary sizes
    src_vocab_size = len(data_processor.src_vocab)
    tgt_vocab_size = len(data_processor.tgt_vocab)
    
    # Initialize model
    print(f"Initializing {model_size} transformer model...")
    model = TransformerModel(config, src_vocab_size, tgt_vocab_size)
    
    # Initialize evaluator
    evaluator = Evaluator(config, model, data_processor)
    
    # If num_checkpoints is not specified, use from config
    if num_checkpoints is None:
        num_checkpoints = config.average_checkpoints
    
    # Find the latest checkpoints
    checkpoint_paths = evaluator.find_latest_checkpoints(checkpoint_dir, num_checkpoints)
    
    if not checkpoint_paths:
        print(f"No checkpoints found in {checkpoint_dir}")
        return
    
    # Average checkpoints
    print(f"Averaging {len(checkpoint_paths)} checkpoints...")
    evaluator.average_checkpoints(checkpoint_paths)
    
    # Save averaged model
    print(f"Saving averaged model to {output_path}...")
    # Create a simple checkpoint structure
    checkpoint = {
        'model': model.state_dict(),
        'epoch': 0,  # Not relevant for averaged model
        'step': 0,   # Not relevant for averaged model
        'loss': 0.0  # Not relevant for averaged model
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the model
    torch.save(checkpoint, output_path)
    
    print(f"Averaged model saved to {output_path}")


def main() -> None:
    """
    Parse command line arguments and run the appropriate function.
    """
    parser = argparse.ArgumentParser(description='Train and evaluate Transformer models')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--model-size', choices=['base', 'big'], default='base',
                             help='Model size (base or big)')
    train_parser.add_argument('--language-pair', choices=['en_de', 'en_fr'], default='en_de',
                             help='Language pair to train on')
    train_parser.add_argument('--config-path', type=str, default=None,
                             help='Path to configuration file')
    train_parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                             help='Directory to save checkpoints')
    train_parser.add_argument('--resume', type=str, default=None,
                             help='Path to checkpoint to resume training from')
    train_parser.add_argument('--epochs', type=int, default=None,
                             help='Number of epochs to train (if None, will use steps from config)')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a model')
    eval_parser.add_argument('model_path', type=str,
                            help='Path to trained model')
    eval_parser.add_argument('--model-size', choices=['base', 'big'], default='base',
                            help='Model size (base or big)')
    eval_parser.add_argument('--language-pair', choices=['en_de', 'en_fr'], default='en_de',
                            help='Language pair to evaluate on')
    eval_parser.add_argument('--config-path', type=str, default=None,
                            help='Path to configuration file')
    eval_parser.add_argument('--average', action='store_true',
                            help='Average checkpoints before evaluation')
    eval_parser.add_argument('--checkpoint-dir', type=str, default=None,
                            help='Directory containing checkpoints (for averaging)')
    eval_parser.add_argument('--num-checkpoints', type=int, default=None,
                            help='Number of checkpoints to average')
    eval_parser.add_argument('--output-file', type=str, default=None,
                            help='Path to write translations to')
    
    # Translate command
    translate_parser = subparsers.add_parser('translate', help='Translate a sentence')
    translate_parser.add_argument('model_path', type=str,
                                help='Path to trained model')
    translate_parser.add_argument('sentence', type=str,
                                help='Sentence to translate')
    translate_parser.add_argument('--model-size', choices=['base', 'big'], default='base',
                                help='Model size (base or big)')
    translate_parser.add_argument('--language-pair', choices=['en_de', 'en_fr'], default='en_de',
                                help='Language pair to translate')
    translate_parser.add_argument('--config-path', type=str, default=None,
                                help='Path to configuration file')
    
    # Average command
    average_parser = subparsers.add_parser('average', help='Average model checkpoints')
    average_parser.add_argument('checkpoint_dir', type=str,
                              help='Directory containing checkpoints')
    average_parser.add_argument('output_path', type=str,
                              help='Path to save the averaged model')
    average_parser.add_argument('--model-size', choices=['base', 'big'], default='base',
                              help='Model size (base or big)')
    average_parser.add_argument('--language-pair', choices=['en_de', 'en_fr'], default='en_de',
                              help='Language pair')
    average_parser.add_argument('--config-path', type=str, default=None,
                              help='Path to configuration file')
    average_parser.add_argument('--num-checkpoints', type=int, default=None,
                              help='Number of checkpoints to average')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if a command was specified
    if args.command is None:
        parser.print_help()
        return
    
    # Run the appropriate function
    if args.command == 'train':
        train_model(
            model_size=args.model_size,
            language_pair=args.language_pair,
            config_path=args.config_path,
            checkpoint_dir=args.checkpoint_dir,
            resume_checkpoint=args.resume,
            epochs=args.epochs
        )
    elif args.command == 'evaluate':
        evaluate_model(
            model_path=args.model_path,
            model_size=args.model_size,
            language_pair=args.language_pair,
            config_path=args.config_path,
            is_averaged=args.average,
            checkpoint_dir=args.checkpoint_dir,
            num_checkpoints=args.num_checkpoints,
            output_file=args.output_file
        )
    elif args.command == 'translate':
        translation = translate(
            model_path=args.model_path,
            sentence=args.sentence,
            model_size=args.model_size,
            language_pair=args.language_pair,
            config_path=args.config_path
        )
        print(f"Translation: {translation}")
    elif args.command == 'average':
        average_model_checkpoints(
            checkpoint_dir=args.checkpoint_dir,
            output_path=args.output_path,
            model_size=args.model_size,
            language_pair=args.language_pair,
            config_path=args.config_path,
            num_checkpoints=args.num_checkpoints
        )


if __name__ == "__main__":
    # Record start time
    start_time = time.time()
    
    # Run main function
    main()
    
    # Print elapsed time
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
