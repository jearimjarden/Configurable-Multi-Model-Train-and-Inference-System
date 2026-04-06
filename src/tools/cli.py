import argparse


def cli_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logger",
        "-l",
        required=False,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        type=str.lower,
    )
    data_parsed = parser.parse_args()

    return data_parsed
