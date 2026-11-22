"""Titik masuk untuk PyInstaller. Tanpa argumen membuka GUI."""
import multiprocessing
import sys

from justpredict.cli import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main(sys.argv[1:]))
