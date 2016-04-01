import sys


def main(f, t, fileout):
    with open(fileout, 'w') as file_:
        for file in range(f, t, 1):
            try:
                file_.write(str(file) + '\n')
            except Exception as e:
                print("Error")


if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
