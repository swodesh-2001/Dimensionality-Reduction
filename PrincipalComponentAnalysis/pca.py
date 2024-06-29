import numpy as np
from cv2 import imread,cvtColor,COLOR_BGR2RGB
import os
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings('ignore') 



class Compressor():
    '''
    A class for compressing image
    '''

    def __init__(self,eigen_no):
        ''''
        Constructor for the compressor class

        parameters
        -------
        eigen_no : int
            N largest eigen vectors to be included in feature matrix
        
        '''
        self.eigen_no = eigen_no
        
    def compress(self,image_path):
        ''''
        Compresses image based on the given number of eigen vector to be included

        parameters
        -------
        image_path : str
            Path of the image to be passed for the image compression
        
        '''
        image = imread(image_path)
        image = cvtColor(image,COLOR_BGR2RGB)
        channels = []

        for i in range(0,image.shape[-1]):
            channel_img = image[:,:,i]
            channel_cov =  np.cov(channel_img)
            channel_eigen_vals, channel_eigen_vectors = np.linalg.eig(channel_cov)
            channel_feature_matrix = channel_eigen_vectors[:,:self.eigen_no]
            pca_channel =   np.dot(channel_img.T , channel_feature_matrix)
            channel_restored_image = np.dot(channel_feature_matrix , pca_channel.T) 
            channel_restored_image = channel_restored_image.astype(np.uint8)
            channels.append(channel_restored_image)

        compressed_image = np.stack((channels[0], channels[1],channels[2]), axis=2)
        compressed_image = compressed_image.astype(np.uint8)
        plt.title('Image Compression inlcuding \n {} largest eigen vectors'.format(self.eigen_no))
        plt.imshow(compressed_image)

    def show_compression(self,image_path):
        '''
        Shows the image compression progression

        parameters
        -------
        image_path : str
            Path of the image to be passed for the image compression
        
        
        '''
        image = imread(image_path)
        image = cvtColor(image,COLOR_BGR2RGB)
        # Number of rows and columns for the grid 
        eigen_to_display = len(range (1,self.eigen_no + 1 , max(1,int(self.eigen_no/10)) ))
        num_rows = eigen_to_display//5 + int((eigen_to_display%5 != 0))
        num_cols = 5
        # Create a figure and a grid of subplots
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 2), squeeze= False)
        row,col = 0,0



        for eigen in range (1,self.eigen_no+1, max(1,int(self.eigen_no/10))):
            channels = []
            for i in range(0,image.shape[-1]):
                channel_img = image[:,:,i]
                channel_cov =  np.cov(channel_img)
                channel_eigen_vals, channel_eigen_vectors = np.linalg.eig(channel_cov)
                channel_feature_matrix = channel_eigen_vectors[:,:eigen]
                pca_channel =   np.dot(channel_img.T , channel_feature_matrix)
                channel_restored_image = np.dot(channel_feature_matrix , pca_channel.T) 
                channel_restored_image = channel_restored_image.astype(np.uint8)
                channels.append(channel_restored_image)

            compressed_image = np.stack((channels[0], channels[1],channels[2]), axis=2)
            compressed_image = compressed_image.astype(np.uint8)

            axs[row][col].imshow(compressed_image)
            axs[row][col].set_title('Inlcuded {} \n largest eigen vectors \n in feature matrix'.format(eigen))
            axs[row][col].axis('off')  # Hide the axes
            col += 1
            if col == num_cols :
                col = 0
                row += 1
 
        plt.tight_layout()
        plt.show()

    
    

