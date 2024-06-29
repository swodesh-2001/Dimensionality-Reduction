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

    def __init__(self,rank):
        ''''
        Constructor for the compressor class

        parameters
        -------
        rank : int
            matrix upto given rank to be approximated for the original image
        
        '''
        self.ranks = rank
    
    def custom_svd(self,A):
        '''
        This is the self implemented svd, the accuracy of this algorithm isn't near to what numpy offers.
        
        returns
        -------
        Decomposed matrix of the Matrix A
        
        '''

 
        AT_A = np.dot(A.T, A).astype(np.float32)
         
        eigenvalues, V = np.linalg.eig(AT_A)
         
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.abs(eigenvalues[sorted_indices])
        V = V[:, sorted_indices]
        
        # Step 4: Compute singular values (square root of eigenvalues)
        singular_values = np.sqrt(np.abs(eigenvalues))
        
        # Step 5: Compute U
        U = np.zeros((A.shape[0], A.shape[0]))
        for i in range(len(singular_values)):
            U[:, i] = np.dot(A, V[:, i]) / singular_values[i]
        
        # Form the Sigma matrix
        Sigma = np.zeros((A.shape[0], A.shape[1]))
        np.fill_diagonal(Sigma, singular_values)
        
        return U, Sigma.sum(axis = 0), V.T



    def compress(self,image_path):
        ''''
        Compresses image based on the given rank

        parameters
        -------
        image_path : str
            Path of the image to be passed for the image compression
        
        '''
        image = imread(image_path)
        image = cvtColor(image,COLOR_BGR2RGB)
        channels = []
        for i in range(0,image.shape[-1]):
            U,S,VT =  np.linalg.svd(image[:,:,i])
            # U,S,VT = self.custom_svd(image[:,:,i])
            compressed_channel = U[:, :self.ranks] @ np.diag(S[:self.ranks])@VT[:self.ranks, :] 
            channels.append(compressed_channel)

        compressed_image = np.stack((channels[0], channels[1],channels[2]), axis=2)
        compressed_image = compressed_image.astype(np.uint8)
        plt.title('Image Compression with Rank {}'.format(self.ranks))
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
        ranks_to_display = len(range (1,self.ranks+1 , max(1,int(self.ranks/10)) ))

        num_rows = ranks_to_display//5 + int((ranks_to_display%5 != 0))
        num_cols = 5
        # Create a figure and a grid of subplots
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 2),squeeze= False)
        row,col = 0,0

        for rank in range (1,self.ranks+1, max(1,int(self.ranks/10))):
            channels = []
            for i in range(0,image.shape[-1]):
                U,S,VT =  np.linalg.svd(image[:,:,i])
                # U,S,VT = self.custom_svd(image[:,:,i])

                compressed_channel = U[:, :rank] @ np.diag(S[:rank])@VT[:rank, :] 
                channels.append(compressed_channel)

            compressed_image = np.stack((channels[0], channels[1],channels[2]), axis=2)
            compressed_image = compressed_image.astype(np.uint8)  

            axs[row][col].imshow(compressed_image)
            axs[row][col].set_title(f'Compressed Image Rank {rank}')
            axs[row][col].axis('off')  # Hide the axes
            col += 1
            if col == num_cols :
                col = 0
                row += 1
 
        plt.tight_layout()
        plt.show()

    
    

