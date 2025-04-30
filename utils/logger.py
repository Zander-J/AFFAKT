import logging
import os


def get_logger(args, distributed_rank):
    output_dir = args.output_dir
    log_file = os.path.join(output_dir, "log.log")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    if distributed_rank > 0:
        return logger
    formatter = logging.Formatter(
        "\n%(asctime)s - %(levelname)s: - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.write_to_local:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        fh.terminator = ""

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    ch.terminator = ""

    logger.addHandler(ch)
    if args.write_to_local:
        logger.addHandler(fh)
    return logger
