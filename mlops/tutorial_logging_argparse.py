import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def go(args):
    logger.info("Starting test script")
    logger.info(f"Hello {args.name}!")
    logger.info(f"Optional argument: {args.optional_arg}")
    logger.warning("This is a warning message")
    logger.error("This is an error message")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test script')  # description is for --help
    parser.add_argument('--name', type=str, help='Name', required=True)
    parser.add_argument(
        '--optional_arg', type=int, help='optional test argument', required=False, default=0)
    test_args = parser.parse_args()
    go(test_args)
