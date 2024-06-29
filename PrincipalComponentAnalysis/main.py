from pca import Compressor
import argparse


if __name__ == '__main__' :
    import argparse
    parser = argparse.ArgumentParser(
        prog = " Image Compressor " ,
        description = "Compresses the image using Principal Component Analysis (PCA) "
    )
    parser.add_argument("-n","--eigen", type=int , help= "Provide the bo of largest eigen vectors to be included for the image compression")
    parser.add_argument("--path", type=str, help= "Provide the path of the image file.")
    args = parser.parse_args()
    compressor = Compressor(args.eigen)
    compressor.compress(args.path)
    compressor.show_compression(args.path)







