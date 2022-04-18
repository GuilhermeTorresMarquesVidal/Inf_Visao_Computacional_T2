import numpy as np
from scipy.spatial import distance

def gradient_orientation(image):
    
    fdx = np.gradient(image, axis = 0)
    fdy = np.gradient(image, axis = 1)
    
    theta = np.arctan(fdx/fdy)
    
    return theta

def theta_transform(theta):
    
    return theta + np.pi/2 

def theta_counts(image_theta):
    return np.unique(
        image_theta[np.isnan(image_theta) == False], 
        return_counts = True)

def counts_norm(counts):
    return counts/np.sum(counts)

def query(sample, count_norm_db, n_queries = 6):
    
    theta = theta_transform(gradient_orientation(sample))
    _, theta_count = theta_counts(theta)
    count_norm = counts_norm(theta_count)
    
    dist = []
    
    for count_norm_sample in count_norm_db:
        dist.append(distance.euclidean(count_norm,count_norm_sample))
    
    return np.argsort(dist)[:n_queries]