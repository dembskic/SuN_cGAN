# (Ex,Eg) unfolding with cGANs
This repository contains a Pix2Pix cGAN trained to unfold (Ex,Eg) matrices produced by the SuN Detector, a segmented NaI(Tl) scintillator developed at the National Superconducting Cyclotron Laboratory. 

Our cGAN implementation is constructed with Tensorflow and based on the work of Isola, Zhu, Zhou, and Efros (https://arxiv.org/abs/1611.07004), as well as Tensorflow's "pix2pix: Image Translation with a Conditional GAN" tutorial (https://www.tensorflow.org/tutorials/generative/pix2pix).

Further information can be found in Dembski, Kuchera, Liddck, Spyrou, and Ramanujan, Two-Dimensional Total Absorption Spectroscopy with Conditional Generative Adversarial Networks. 



# Version Dependencies 

This code is intended to be run with Python3, with a single C++ file used to convert ROOT data output by the GEANT4 simulation package to CSV files. Required modules and the versions utilized are listed below. 


numpy - version 1.21.3
matplotlib - version 3.3.3
scikit-learn - version 1.0.1
pandas - version 0.25.0
seaborn - version 0.11.2 (optional for viz style purposes)


These modules can be installed via pip:

```

pip install numpy matplotlib scikit-learn seaborn

```

The model is created with 

tensorflow - verson 2.3.1

It is recommended that GPU-compatible Tensorflow is installed to expedite the training time. This can be done via 

```

pip install tensorflow-gpu==2.3.1

```

  


# Usage Notes

This implementation relies on the following module and function imports: 

```
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

```

Additionally, it is recommended to set Tensorflow's INFO and WARNING messages via the following command:

```

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

```

The generator and discriminator models are initialized via


```

generator=Generator()
discriminator=Discriminator()

```

The generator is an instance of the UNet architecture and the discriminator utilizes a PatchGAN architecture, both as originally utilized by Isola et al. The networks are initialized for an input shape of [512,512,1]. The generator's output layer includes a Sigmoid activation function. Desired edits to input size or other aspects of model construction can be done in cGAN.py.

To train the model, call 

```

fit(train_dataset, epochs, test dataset)

```

Where 

- train_dataset is a batched Tensorflow Dataset containing training spectra and labels 
- epochs is an integer representing the number of times to iterate through the data during training
- test_dataset is a batched Tensorflow Dataset containing testing spectra and labels

Tensorflow Datasets can be created from data stored in Numpy arrays as follows:

```

train_data=np.load("/path/to/data/train_data.npy")
train_labels=np.load("/path/to/labels/train_labels.npy")
test_data=np.load("/path/to/data/test_data.npy")
test_labels=np.load("/path/to/labels/test_labels.npy")

train_dataset = tf.data.Dataset.from_tensor_slices((np.float32(data_train), np.float32(labels_train)))
test_dataset=tf.data.Dataset.from_tensor_slices((np.float32(data_test), np.float32(labels_test)))


train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size)
test_dataset = test_dataset.shuffle(buffer_size).batch(batch_size)

```
With batch_size and buffer_size as integer variables. 

An additional method of creating Tensorflow Datasets via Python generators can be implemented buy running

```
from TFGenerator import Get_Data()

train_dataset, test_dataset=Get_Data()

```
See train_example.py for an example that loads, train/test splits, and standardizes data stored in a .npz file. 

Following training, model analysis can be done using the KMeans framework. It is recommended to divide the testing dataset into arrays of single and double-gamma ray examples for simplicity. 

```

singles_predictions=generator(singles_test).numpy()[:,:,:,0] #final shape: (bs,512,512)
doubles_predictions=generator(doubles_test).numpy()[:,:,:,0] #final shape: (bs,512,512)

```

The KMeans analysis is called via 

```

singles_centroid_errs, singles_error_ratios, singles_energies=single_centroid_calc(singles_predictions, singles_labels, threshold)
doubles_energies, doubles_error_ratios, doubles_log_ratios=double_centroid_calc(doubles_predictions, doubles_labels, threshold)

doubles_energies, doubles_error_ratios, doubles_log_ratios=double_centroid_calc(doubles_predictions, doubles_labels, threshold)

```

Where threshold is a decimal variable. Any pixels in the prediction spectra below threshold are rounded to zero to eliminate low-value noise resulting from the sigmoid function. 


For a comprehensive example that loads and processes simulated (Ex,Eg) matrices, trains a model, and performs KMeans analysis, run train_example.py via 

```

python train_example.py

```

train_example.py will train a Pix2Pix model on a small subset of our training data, provided in the DataFiles folder, and analyze model testing results via the KMeans analysis framework. A successful run will output the following:

-training_checkpoints, a directory of model checkpoints from throughout the training
-logs, a directory of timestamped Tensorflow events files, which can be loaded into Tensorboard to visualize loss curves for the four losses used by the model
-singles_analysis.png, a 2D histogram showing percent error results for the provided single gamma-ray testing spectra
-doubles_analysis.png, a 2D histogram showing percent error results for the provided double gamma-ray testing spectra

Our best weights can also be loaded into the model to view results from a training with a larger training dataset.
