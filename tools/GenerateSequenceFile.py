import sys


def main(f, t, l, fileout):
    with open(fileout, 'w') as file_:
        for z in range(0, l, 1):
            for file in range(f, t, 1):
                try:
                    file_.write(str(file) + '\n')
                except Exception as e:
                    print("Error")


if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
