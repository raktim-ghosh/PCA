import numpy
from numpy import linalg as la
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
 
gdal.UseExceptions()
 
 
class PCA(object):
    def __init__(self, image):
        '''  Image has been put into try/except block to check whether it is readable element or not  '''
        self.__band_list = []
        self.__mean_vec = []
        self.__pic_vec = []
        self.__cov_mat = None
        self.__eig_val = None
        self.__eig_vec = None
        try:
            ds = gdal.Open(image)
            if ds:
                self.__image = image
                assert isinstance(ds, object)
                self.__ds = ds
        except RuntimeError as err:
            print("Exception", err)
 
    def get_bands(self):
        ''' Band has been extracted from main image using gdal library '''
        print('Extracting Bands')
        obj_img = self.get_ds()
        for band in range(1, obj_img.RasterCount + 1):
            img_band = self.get_ds().GetRasterBand(band).ReadAsArray().astype(float)
            self.__band_list.append(img_band)
            self.__mean_vec.append(np.mean(img_band))
 
    def get_vec(self):
        ''' Each image pixel has been converted into vector '''
        print('Getting pixel vectors of an image')
        vec = []
        for band in self.get_band_list():
            band = band.reshape(band.shape[0] * band.shape[1])
            vec.append(band.tolist())
 
        for values in zip(*vec):
            self.__pic_vec.append(values)
        return self.__pic_vec
 
    def get_cov_mat(self):
        ''' Covariance matrix has been generated of entire image '''
        print('Getting covariance matrix of an image')
        s = 0
        pvc = self.get_pic_vec()
        for values in pvc:
            cov = np.matrix(values) - np.matrix(self.get_mean_vec())
            cov = cov.T * cov
            s += cov
        self.__cov_mat = np.array(s) / len(pvc)
        print(self.__cov_mat)
        return self.__cov_mat
 
    def get_eig_vec(self):
        ''' From the above covariance matrix, eigenvalues and eigenvectors has been computed '''
        info_content = []
        print('Getting Eigenvalues of an Image')
        cov_mat = self.get_covariance()
        w, v = la.eig(cov_mat)
        self.__eig_val = np.abs(w)
        for i in range(len(self.__eig_val.tolist())):
            k = self.__eig_val.tolist()[i]*100 / sum(self.__eig_val.tolist())
            info_content.append(k)
        print(info_content)
        plt.plot(range(1, self.get_ds().RasterCount+1), info_content, '-ro')
        plt.plot(range(1, self.get_ds().RasterCount+1), self.__eig_val/max(self.__eig_val), '-ro')
#        plt.scatter(range(1, self.get_ds().RasterCount + 1), self.__eig_val / max(self.__eig_val), 'r')
        plt.xlim([1, 11])
#        plt.axis(0, 250, 0, 1)
        plt.show()
        self.__eig_vec = np.abs(v) ** 2
        print('EIGENVALUE COMPLETE!!')
        return self.__eig_vec
 
    def get_pca(self):
        ''' The principal component analysis has been performed '''
        self.get_bands()
        self.get_vec()
        self.get_cov_mat()
        self.get_eig_vec()
        print('Extracting principal components of an image')
        pvc = self.get_pic_vec()
        eig_vec = self.get_eig_vec_img()
        pca_pvc_val = []
        for value in pvc:
            val = np.matrix(value)
            pca_pvc_val.append(eig_vec.T * val.T)
 
        pca_pvc_val_mf = []
        for x in pca_pvc_val:
            temp = x.tolist()
            flat_list = [item for sublist in temp for item in sublist]
            pca_pvc_val_mf.append(flat_list)
 
        img_as_lst = []
        for values in zip(*pca_pvc_val_mf):
            img_as_lst.append(values)
 
        for (index, image) in enumerate(img_as_lst):
            arr = np.array([image[i:i + self.get_ds().RasterXSize]
                            for i in range(0, len(img_as_lst[0]), self.get_ds().RasterXSize)])
            maxval = max(abs(np.min(arr)), np.max(arr))
            arr = arr / maxval
            arr[arr < 0] += np.float(1)
            self.mat_to_image(arr, 'PCA_Outputs/pca_' + str(index+1))
        return pca_pvc_val
 
    @staticmethod
    def mat_to_image(arr_out, outfile):
        cols, rows = arr_out.shape
        driver = gdal.GetDriverByName("GTiff")
        outfile += '.tiff'
        outdata = driver.Create(outfile, rows, cols, 1, gdal.GDT_Float32)
        outdata.GetRasterBand(1).WriteArray(arr_out)
        outdata.FlushCache()
 
    def get_image(self):
        return self.__image
 
    def set_image(self, image):
        self.__image = image
 
    def get_ds(self):
        return self.__ds
 
    def set_ds(self, ds):
        self.__ds = ds
 
    def get_band_list(self):
        return self.__band_list
 
    def get_mean_vec(self):
        return self.__mean_vec
 
    def get_pic_vec(self):
        return self.__pic_vec
 
    def get_eig_vec_img(self):
        return self.__eig_vec
 
    def get_covariance(self):
        return self.__cov_mat
 
 
img = PCA('EO1H1460392009331110PO.L1R')
img.get_pca()
