import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ablation")
logger.setLevel(logging.INFO)


def timing(f,):
    import time

    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()

        logger.info(
            f"{f.__name__} function took {(time2 - time1)/60.0:.3f} min"
        )

        return ret

    return wrap
