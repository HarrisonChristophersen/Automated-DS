"""
Command-line interface for Automated DS
Version: 2.0 - Refactored Architecture
"""
import argparse
import sys
from pathlib import Path

from config import AppConfig, AutoMLConfig, LIMEConfig, PreprocessingConfig
from pipeline import MLPipeline
from logger import setup_logger


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Automated DS - AutoML with Explainability",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py data.csv target
  python main.py sales.xlsx Revenue --max_models 30 --max_secs 600
  python main.py data.csv target --output results/ --exclude date,id
        """
    )
    
    # Required arguments
    parser.add_argument(
        "file",
        help="Path to CSV or Excel file"
    )
    parser.add_argument(
        "target",
        help="Target column name to predict"
    )
    
    # AutoML configuration
    parser.add_argument(
        "--max_models",
        type=int,
        default=20,
        help="Maximum number of AutoML models to train (default: 20)"
    )
    parser.add_argument(
        "--max_secs",
        type=int,
        default=300,
        help="Maximum runtime in seconds (default: 300)"
    )
    parser.add_argument(
        "--nfolds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)"
    )
    
    # LIME configuration
    parser.add_argument(
        "--num_features",
        type=int,
        default=10,
        help="Number of features to show in LIME explanations (default: 10)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of instances to explain with LIME (default: 10)"
    )
    
    # Preprocessing configuration
    parser.add_argument(
        "--missing_threshold",
        type=float,
        default=0.5,
        help="Drop columns with missing % above this (default: 0.5)"
    )
    parser.add_argument(
        "--numeric_strategy",
        choices=['median', 'mean', 'interpolate'],
        default='median',
        help="Strategy for filling numeric missing values (default: median)"
    )
    parser.add_argument(
        "--no_scaling",
        action='store_true',
        help="Disable feature scaling"
    )
    parser.add_argument(
        "--no_feature_engineering",
        action='store_true',
        help="Disable automatic feature engineering"
    )
    
    # Output configuration
    parser.add_argument(
        "--output",
        default="outputs",
        help="Output directory for results (default: outputs)"
    )
    parser.add_argument(
        "--exclude",
        help="Comma-separated list of columns to exclude from training"
    )
    
    # Logging
    parser.add_argument(
        "--log_level",
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--quiet",
        action='store_true',
        help="Suppress all output except errors"
    )
    
    return parser.parse_args()


def create_config(args) -> AppConfig:
    """Create application configuration from arguments"""
    
    automl_config = AutoMLConfig(
        max_models=args.max_models,
        max_runtime_secs=args.max_secs,
        nfolds=args.nfolds
    )
    
    lime_config = LIMEConfig(
        num_features=args.num_features,
        num_samples=args.num_samples
    )
    
    preprocessing_config = PreprocessingConfig(
        missing_threshold=args.missing_threshold,
        numeric_strategy=args.numeric_strategy,
        scale_numeric=not args.no_scaling,
        engineer_features=not args.no_feature_engineering
    )
    
    return AppConfig(
        automl=automl_config,
        lime=lime_config,
        preprocessing=preprocessing_config,
        log_level='ERROR' if args.quiet else args.log_level,
        output_dir=args.output
    )


def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Validate input file
    if not Path(args.file).exists():
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        sys.exit(1)
    
    # Create configuration
    config = create_config(args)
    logger = setup_logger('main', config.log_level)
    
    try:
        logger.info("Starting Automated DS pipeline")
        logger.info(f"Input file: {args.file}")
        logger.info(f"Target column: {args.target}")
        
        # Parse exclude columns
        exclude_cols = None
        if args.exclude:
            exclude_cols = [col.strip() for col in args.exclude.split(',')]
            logger.info(f"Excluding columns: {exclude_cols}")
        
        # Create and run pipeline
        pipeline = MLPipeline(config)
        results = pipeline.run(args.file, args.target, exclude_cols)
        
        # Display report
        if not args.quiet:
            report = pipeline.generate_report(args.target)
            print("\n" + report)
        
        # Save results
        pipeline.save_results(config.output_dir)
        
        logger.info(f"Results saved to {config.output_dir}")
        logger.info("Pipeline completed successfully!")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        return 130
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())