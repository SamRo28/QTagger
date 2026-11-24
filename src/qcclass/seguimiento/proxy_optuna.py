"""Optuna gRPC storage proxy in front of an SQLite database (run by ``qcclass servidores``).

Usage: ``python -m qcclass.seguimiento.proxy_optuna --almacenamiento sqlite:///<db>
--host 0.0.0.0 --puerto 13000``. Only this process opens the database, so SQLite's file
locking is never shared between machines.
"""

import argparse
import logging
import warnings

import optuna

logger = logging.getLogger(__name__)


def main() -> None:
    """Parse the arguments and serve until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--almacenamiento", required=True, help="RDB URL, e.g. sqlite:///x.db")
    parser.add_argument("--host", required=True)
    parser.add_argument("--puerto", type=int, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # get_storage adds the in-process cache recommended by Optuna for the proxy
    warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
    storage = optuna.storages.get_storage(args.almacenamiento)
    logger.info("Optuna gRPC proxy on %s:%d for %s", args.host, args.puerto, args.almacenamiento)
    optuna.storages.run_grpc_proxy_server(storage, host=args.host, port=args.puerto)


if __name__ == "__main__":
    main()
