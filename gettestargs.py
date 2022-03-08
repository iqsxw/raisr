import argparse

def gettestargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filter", help="Use file as filter")
    parser.add_argument("-p", "--plot", help="Visualizing the process of RAISR image upscaling", action="store_true")
    parser.add_argument("-i", "--input", help="Input video file")
    args = parser.parse_args()
    return args
