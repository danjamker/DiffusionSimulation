import sys

import Tools


def main(f, t):
    with open(t, 'w+') as file_:
        for file in Tools.list(f):
            print(file)
            try:
                file_.write(f + "/" + file + '\n')
            except Exception as e:
                print("Error")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
