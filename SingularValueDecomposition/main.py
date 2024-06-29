from svd import Compressor
import argparse


if __name__ == '__main__' :
    import argparse
    parser = argparse.ArgumentParser(
        prog = " Image Compressor " ,
        description = "Compresses the image using  Singular Value Decomposition (SVD) "
    )
    parser.add_argument("-r","--ranks", type=int , help= "Provide the rank for the image compression")
    parser.add_argument("-p","--path", type=str, help= "Provide the path of the image file.")
    args = parser.parse_args()
    compressor = Compressor(args.ranks)
    compressor.compress(args.path)
    compressor.show_compression(args.path)







