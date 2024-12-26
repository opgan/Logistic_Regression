import logging


def log(msg):
    """
    This function records log messages into a file 'info.log'

    Argument:
    msg -- text info

    Returns:
    updated info.log under log folder
    """

    # Configure logging to capture INFO messages and above
    logging.basicConfig(
        filename="log/info.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logging.info(msg)
