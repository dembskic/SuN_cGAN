import numpy as np 

import matplotlib.pyplot as plt 
import matplotlib 

from sklearn.cluster import KMeans

import seaborn 
seaborn.set_style("whitegrid")



#Inputs: Prediction image of size 512x512 (numpy array). corresponding label image (same dimensions, numpy array). Threshold value at which pixels will be considered in analysis. 
#Outputs: Location of max value in prediction image (in energy space). Location of max value in label. xy locations of all pixels above threshold (energy space). Values of these pixels. All numpy arrays.   

def max_points(prediction, label, threshold): 
    
    #Pull locations for max values in prediction image and label 
    #np.where returns a tuple with array indices of max value locations, which are converted to arrays
    max_pred=np.asarray(np.where(prediction == np.max(prediction)))
    true_max=np.asarray(np.where(label == np.max(label)))    
    
    #Convert pixel indices into energy space. Assuming 20 keV per bin and 512x512 images. 
    #Python binning for 2D arrays uses a different scheme than Root, counting from top left instead of bottom. 
    #These differences are accounted for in going to energy space. 
    max_pred=np.asarray((20*max_pred[1], 20*(511-max_pred[0])))
    true_max=np.asarray((20*true_max[1], 20*(511-true_max[0])))
    

    
    #Define empty arrays for xy locations and counts for high-value pixels 
    xys=[]
    values=[]
    
    #Iterate through all pixels in image
    for i in range(len(prediction[0])):
        for j in range(len(prediction[1])):
        
            #Append xy energy space location and value of any pixels with counts above the threshold 
            if prediction[i,j] > threshold:
                xys.append((20*j,20*(511-i)))
                values.append(prediction[i,j])
                
    
    #Accounting for low-intensity images
    #Sometimes, one image in a set will have very low overall counts relative to the others and fall below what 
    #would otherwise be a reasonable threshold 
    #In these cases, its cleaner to take the max value of the outlier location instead of lowering the threshold 
    #for the entire set 
    if xys == []:
    
        x=np.asarray(np.where(prediction == np.max(prediction)))
        x=np.reshape(x, [2])
        x=np.asarray((20*x[1],20*(511-x[0])))
        xys.append(x)
        values.append(np.max(prediction))
      
    #Convert to Numpy Arrays 
    xys=np.asarray(xys)
    values=np.asarray(values)
    
    return max_pred, true_max, xys, values
    
    




#Inputs: Numpy array of predictions (Nx512x512). Numpy array of labels (Nx512x512). Threshold to be passed to max_points
#Outputs: Numpy array of centroid errors per prediction (size N). Numpy array of percent error per prediction. Numpy array of true energy values for each image. 
def single_centroid_calc(predictions, labels, threshold):
    
    #Define empty arrays for quantities to return.
    centroid_diffs=[]
    ratios=[]
    energies=[]

    #Iterate over all images in prediciton array 
    for i in range(len(predictions)):
        
      
        #Call max_points for each image to pull high pixels for KMeans clustering   
        max, maxtrue, xys, values=max_points(predictions[i], labels[i],threshold)
        
        
        
        #Call KMeans and fit to high pixels and their values. n=1 for single gamma rays.
        km = KMeans(n_clusters=1, init='random',n_init=10, max_iter=300, tol=1e-04, random_state=0)
        y_km = km.fit_predict(xys, sample_weight=values)
        
        
        #Calcualte absolute error in energy space between KMeans cluster location and max location in label 
        centroid_diff=np.sqrt((maxtrue[0]-km.cluster_centers_[0,0])**2+(maxtrue[1]-km.cluster_centers_[0,1])**2)


        #Calculate percent error (centroid error/true energy)
        energy_ratio=centroid_diff/maxtrue[0]

        #Append centroid error, percent error, and true energy value
        centroid_diffs.append(centroid_diff)
        ratios.append(energy_ratio)
        energies.append(maxtrue[0])

        print(i, len(xys))
    
    #Convert lists to numpy arrays
    centroid_diffs=np.asarray(centroid_diffs)
    ratios=np.asarray(ratios)
    energies=np.asarray(energies)
    
    #Multiply by 100 to get true percentage
    ratios=ratios*100

    return centroid_diffs, ratios, energies






def peak_errors(maxtrue, cluster_centers_):
            small_peak=np.asarray((maxtrue[0,0], maxtrue[1,0]))
            big_peak=np.asarray(((maxtrue[0,1], maxtrue[1,1])))
            
            
        
            if cluster_centers_[0,0] >= cluster_centers_[1,0]:
    
                big_err=big_peak-cluster_centers_[0]
                small_err=small_peak-cluster_centers_[1]
    
            if cluster_centers_[0,0] < cluster_centers_[1,0]:
    
                big_err=big_peak-cluster_centers_[1]
                small_err=small_peak-cluster_centers_[0]


        
            #Quadrature sum for xy-array centroid errors
            
            large_centroid_err=np.sqrt(big_err[0]**2+big_err[1]**2)
            small_centroid_err=np.sqrt(small_err[0]**2+small_err[1]**2)
        
        
        
            #Calcualte appropriate percent errors
            energy_ratio_big=100*(large_centroid_err/big_peak[0])
            energy_ratio_small=100*(small_centroid_err/small_peak[0])
            
            return large_centroid_err, small_centroid_err, energy_ratio_big, energy_ratio_small





def double_centroid_calc(doubles_preds, doubles_labels, threshold):

    centroid_errs_large=[]
    centroid_errs_small=[]
    energy_ratios_big=[]
    energy_ratios_small=[]
    energies_small=[]
    energies_large=[]




    for i in range(len(doubles_preds)):

        max, maxtrue, xys, values=max_points(doubles_preds[i], doubles_labels[i], threshold)
        print(i, len(xys))
        if maxtrue.shape == (2,1):
            maxtrue=np.asarray((maxtrue, maxtrue))[:,:,0]



        points=np.sort(np.unique(doubles_preds[i]))
        second_highest=points[len(points)-2]

        if second_highest < threshold: 

            

            x=np.asarray(np.where(doubles_preds[i] == second_highest))
            x=np.reshape(np.asarray((20*x[1], 20*(511-x[0]))), [1,2])
            xys=np.concatenate((xys, x), axis=0)
            values=np.append(values, threshold)




        km = KMeans(n_clusters=2, init='random',n_init=10, max_iter=300, tol=1e-04, random_state=0)
        y_km = km.fit_predict(xys, sample_weight=values)


        Es=((maxtrue[0,0],maxtrue[1,0]), (maxtrue[0,1], maxtrue[1,1]))
        energy_big=maxtrue[0,1]
        energy_small=maxtrue[0,0]



        large_centroid_err, small_centroid_err, energy_ratio_large, energy_ratio_small=peak_errors(maxtrue, km.cluster_centers_)





        centroid_errs_large.append(large_centroid_err)
        centroid_errs_small.append(small_centroid_err)
        energy_ratios_big.append(energy_ratio_large)
        energy_ratios_small.append(energy_ratio_small)
        energies_small.append(energy_small)
        energies_large.append(energy_big)

    energy_ratios=np.concatenate((energy_ratios_big, energy_ratios_small), axis=0)
    energies=np.concatenate((energies_large, energies_small), axis=0)

    log_ratios=np.log10(energy_ratios)
    for i in range(len(log_ratios)):
        if log_ratios[i] < 0:
            log_ratios[i] = 0

            
    return energies, energy_ratios, log_ratios


