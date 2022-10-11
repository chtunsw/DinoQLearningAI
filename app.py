import sys
from pathlib import Path

file_dir = Path(__file__).parent

def print_instructions():
    print("Instructions")
    print("train model : python3 app.py train")
    print("test model : python3 app.py test")

if __name__ == "__main__":
    args = sys.argv[1:]
    if args:
        arg = args[0]
        if arg == "train":
            pass
        elif arg == "test":
            pass
        else:
            print_instructions()
    else:
        print_instructions()