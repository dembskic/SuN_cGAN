#Imports
import numpy as np 
import matplotlib 
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split 
from sklearn.utils import shuffle
import sys
import seaborn 
seaborn.set_style("whitegrid")

sys.path.insert(0,'src')

from cGAN import Generator, Discriminator, generator_loss, discriminator_loss, train_step, fit
from Analysis import max_points, single_centroid_calc, peak_errors, double_centroid_calc

#Disable Tensorflow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


###########Data processing############################

spectra=np.load("data/spectra.npz")

#Parse singles and doubles
singles=spectra["singles"]
singles_labels=spectra["singles_labels"]
doubles=spectra["doubles"]
doubles_labels=spectra["doubles_labels"]

#Split singles/doubles into train, test data
singles_train, singles_test, singles_labels_train, singles_labels_test=train_test_split(singles, singles_labels, test_size=.2)
doubles_train, doubles_test, doubles_labels_train, doubles_labels_test=train_test_split(doubles, doubles_labels, test_size=.2)


#Concatenate into single train, test set (in test set, first half will be singles and second half will be doubles)

data_train=np.concatenate((singles_train, doubles_train), axis=0)
data_test=np.concatenate((singles_test, doubles_test), axis=0)

labels_train=np.concatenate((singles_labels_train, doubles_labels_train), axis=0)
labels_test=np.concatenate((singles_labels_test, doubles_labels_test), axis=0)


#Standardize train and test datasets to zero mean and unit variance 

data_train=(data_train-np.mean(data_train))/np.std(data_train)
data_test=(data_test-np.mean(data_test))/np.std(data_test)

print(data_train.shape, data_test.shape)
#Create Tensorflow Dataset structures from Numpy Arrays

buffer_size = 4
batch_size = 5

#Dtype=float32
train_dataset = tf.data.Dataset.from_tensor_slices((np.float32(data_train), np.float32(labels_train)))
test_dataset=tf.data.Dataset.from_tensor_slices((np.float32(data_test), np.float32(labels_test)))

#Shuffle and batch Datasets
train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size)
test_dataset = test_dataset.shuffle(buffer_size).batch(batch_size)


###########Training############################
epochs=1

print("Training for", epochs, "epochs")

generator=Generator()
discriminiator=Discriminator()

fit(train_dataset, epochs, test_dataset)

#Optional: Save weights
#generator.save_weights("data/training_weights.h5")

print("Done Training!")

###########Analysis############################


#Optional: Load weights from another training
#generator.load_weights("data/generator_weights.h5")


#Re-divide test arrays into singles and doubles, post-standardization
length=len(data_test)

singles_test=data_test[0:int(length/2)]
singles_test_labels=labels_test[0:int(length/2)]

doubles_test=data_test[int(length/2):length]
doubles_test_labels=labels_test[int(length/2):length]

print(singles_test.shape, singles_test_labels.shape, doubles_test.shape, doubles_test_labels.shape)

#Generator Predictions

print("Predicting")

singles_predictions=generator(singles_test)
doubles_predictions=generator(doubles_test)

singles_predictions=singles_predictions.numpy()[:,:,:,0]
doubles_predictions=doubles_predictions.numpy()[:,:,:,0]

print(singles_predictions.shape, doubles_predictions.shape)

#Set KMeans Cutoff Threshold
threshold=.2

print("Analyzing Singles")
singles_centroid_errs, singles_error_ratios, singles_energies=single_centroid_calc(singles_predictions, singles_labels, threshold)

singles_energies=singles_energies.reshape((len(singles_energies)))
singles_error_ratios=singles_error_ratios.reshape((len(singles_error_ratios)))

print("Analyzing Doubles")
doubles_energies, doubles_error_ratios, doubles_log_ratios=double_centroid_calc(doubles_predictions, doubles_labels, threshold)


#Doubles Figure
plt.hist2d(doubles_energies, doubles_error_ratios, cmap='Purples', bins=50)
plt.grid(False)
plt.yscale('log')
plt.colorbar()
plt.title("Relative Error by Energy Value (Doubles)")
plt.xlim(0,10000)
plt.xlabel("Energy")
plt.ylabel("Percent Error")
plt.savefig("doubles_analysis.png")

plt.clf()

#Singles Figure
plt.hist2d(singles_energies, singles_error_ratios, cmap='Purples', bins=100)
plt.grid(False)
plt.yscale('log')
plt.colorbar()
plt.xlim(0,10000)
plt.title("Relative Error by Energy Value (Singles)")
plt.xlabel("Energy")
plt.ylabel("Percent Error")
plt.savefig("singles_analysis.png")

print("Finished")
