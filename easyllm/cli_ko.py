from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description="프로그램 설명")
    parser.add_argument("-f", "--foo", help="foo 인수에 대한 설명", required=True)

    return parser.parse_args()


def main():
    args = parse_args()
    print(args)
