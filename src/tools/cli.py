import argparse


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logger",
        "-l",
        required=False,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        type=str.lower,
    )
    parsed_data = parser.parse_args()

    return parsed_data
