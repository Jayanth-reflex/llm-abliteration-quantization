"""
Main entry point for LLM Toolkit CLI.
Provides unified command-line interface for all toolkit functionality.
"""

import argparse
import sys
import logging
from typing import List, Optional

from .quantization import QuantizationCLI
from .abliteration import AbliterationCLI
from .multimodal import MultiModalOptimizer
from .distributed import DistributedQuantizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog='llm_toolkit',
        description='Production-ready toolkit for LLM optimization and modification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quantize a model using QLoRA
  python -m llm_toolkit quantize --model llama2-7b --bits 4 --method qlora
  
  # Abliterate a model with specific strength
  python -m llm_toolkit abliterate --model llama2-7b --strength 0.8
  
  # Optimize a multimodal model
  python -m llm_toolkit multimodal --model clip-vit-base --optimize vision
  
  # Distributed quantization across GPUs
  python -m llm_toolkit distributed --model llama2-13b --gpus 2 --strategy tensor_parallel
        """
    )
    
    parser.add_argument(
        '--version', 
        action='version', 
        version='%(prog)s 1.0.0'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands',
        metavar='COMMAND'
    )
    
    # Quantization subcommand
    quant_parser = subparsers.add_parser(
        'quantize',
        help='Quantize language models using various methods',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quantization Methods:
  qlora    - QLoRA: Efficient finetuning with 4-bit quantization
  gptq     - GPTQ: GPU-based post-training quantization  
  awq      - AWQ: Activation-aware weight quantization
  smoothquant - SmoothQuant: Smooth activation quantization
  int8     - LLM.int8(): 8-bit inference without degradation
        """
    )
    
    quant_parser.add_argument(
        '--model', '-m',
        required=True,
        help='Model name or path (HuggingFace format)'
    )
    
    quant_parser.add_argument(
        '--method',
        choices=['qlora', 'gptq', 'awq', 'smoothquant', 'int8'],
        default='qlora',
        help='Quantization method to use (default: qlora)'
    )
    
    quant_parser.add_argument(
        '--bits',
        type=int,
        choices=[2, 3, 4, 8],
        default=4,
        help='Number of quantization bits (default: 4)'
    )
    
    quant_parser.add_argument(
        '--output', '-o',
        help='Output directory for quantized model'
    )
    
    quant_parser.add_argument(
        '--calibration-data',
        help='Path to calibration dataset (JSON file)'
    )
    
    quant_parser.add_argument(
        '--group-size',
        type=int,
        default=128,
        help='Group size for quantization (default: 128)'
    )
    
    # Abliteration subcommand
    abl_parser = subparsers.add_parser(
        'abliterate',
        help='Remove refusal behaviors from language models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Abliteration Techniques:
  inference    - Runtime intervention during inference
  orthogonal   - Permanent weight orthogonalization
  selective    - Target-specific abliteration
  combined     - Multiple techniques combined
        """
    )
    
    abl_parser.add_argument(
        '--model', '-m',
        required=True,
        help='Model name or path (HuggingFace format)'
    )
    
    abl_parser.add_argument(
        '--strength',
        type=float,
        default=0.8,
        help='Abliteration strength (0.0-1.0, default: 0.8)'
    )
    
    abl_parser.add_argument(
        '--method',
        choices=['inference', 'orthogonal', 'selective', 'combined'],
        default='inference',
        help='Abliteration method (default: inference)'
    )
    
    abl_parser.add_argument(
        '--output', '-o',
        help='Output directory for abliterated model'
    )
    
    abl_parser.add_argument(
        '--refusal-data',
        help='Path to refusal dataset for direction computation'
    )
    
    abl_parser.add_argument(
        '--target-topics',
        nargs='+',
        help='Specific topics for selective abliteration'
    )
    
    # Multimodal subcommand
    mm_parser = subparsers.add_parser(
        'multimodal',
        help='Optimize multimodal models (vision-language)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported Models:
  clip         - CLIP (Contrastive Language-Image Pre-training)
  blip2        - BLIP-2 (Bootstrapped Vision-Language Pre-training)
  llava        - LLaVA (Large Language and Vision Assistant)
  flamingo     - Flamingo (Few-shot learning for multimodal tasks)
        """
    )
    
    mm_parser.add_argument(
        '--model', '-m',
        required=True,
        help='Multimodal model name or path'
    )
    
    mm_parser.add_argument(
        '--optimize',
        choices=['vision', 'language', 'both'],
        default='both',
        help='Components to optimize (default: both)'
    )
    
    mm_parser.add_argument(
        '--vision-bits',
        type=int,
        default=8,
        help='Vision encoder quantization bits (default: 8)'
    )
    
    mm_parser.add_argument(
        '--language-bits',
        type=int,
        default=4,
        help='Language model quantization bits (default: 4)'
    )
    
    mm_parser.add_argument(
        '--output', '-o',
        help='Output directory for optimized model'
    )
    
    # Distributed subcommand
    dist_parser = subparsers.add_parser(
        'distributed',
        help='Distributed quantization across multiple GPUs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Distribution Strategies:
  tensor_parallel  - Split tensors across GPUs
  pipeline_parallel - Split layers across GPUs  
  hybrid          - Combination of tensor and pipeline parallelism
  data_parallel   - Replicate model, split data
        """
    )
    
    dist_parser.add_argument(
        '--model', '-m',
        required=True,
        help='Model name or path'
    )
    
    dist_parser.add_argument(
        '--gpus',
        type=int,
        required=True,
        help='Number of GPUs to use'
    )
    
    dist_parser.add_argument(
        '--strategy',
        choices=['tensor_parallel', 'pipeline_parallel', 'hybrid', 'data_parallel'],
        default='tensor_parallel',
        help='Distribution strategy (default: tensor_parallel)'
    )
    
    dist_parser.add_argument(
        '--bits',
        type=int,
        default=4,
        help='Quantization bits (default: 4)'
    )
    
    dist_parser.add_argument(
        '--output', '-o',
        help='Output directory for distributed model'
    )
    
    return parser

def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    
    # Configure logging
    if parsed_args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Handle no command case
    if not parsed_args.command:
        parser.print_help()
        return 1
    
    try:
        # Route to appropriate handler
        if parsed_args.command == 'quantize':
            cli = QuantizationCLI()
            return cli.run(parsed_args)
        
        elif parsed_args.command == 'abliterate':
            cli = AbliterationCLI()
            return cli.run(parsed_args)
        
        elif parsed_args.command == 'multimodal':
            optimizer = MultiModalOptimizer()
            return optimizer.run(parsed_args)
        
        elif parsed_args.command == 'distributed':
            quantizer = DistributedQuantizer()
            return quantizer.run(parsed_args)
        
        else:
            logger.error(f"Unknown command: {parsed_args.command}")
            return 1
    
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 130
    
    except Exception as e:
        logger.error(f"Error: {e}")
        if parsed_args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())