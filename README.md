# Artist identification from their Paintings.
# Data Science Life Cycles:
# STEP-1 Purpose, motivation and description:

1. Artist identification is traditionally performed by art historians and curators who have expertise and familiarity with different artists and styles of art. This is a complex and interesting problem for computers because identifying an artist does not just require object or face detection; artists can paint a wide variety of objects and scenes. 

2. Additionally, many artists from the same time period will have similar styles, and some such as Pablo Picasso  have painted in multiple styles and changed their style over time. Instead of hand-crafting features, we train CNNs for this problem. 

3. This approach is motivated by the hypothesis that every artist has their own unique style of art and that we can improve upon existing artist identification methods by using a CNN to determine the best possible feature representation of paintings.

4. Our dataset consists of 300 paintings per artist from 57 well-known artists. We train a variety of models ranging from a simple CNN designed from scratch to a ResNet-18 network with transfer learning. Our best network achieves significantly higher classification accuracy than prior work. 

5. Additionally, we perform multiple experiments to explore and understand the learned representation of our networks. Our results demonstrate that CNNs are a powerful tool for artist identification and that when asked to identify artists, they are able to learn a representation of painting style.

6. Our document has two key contributions: 
   •	Train a neural network that significantly outperforms existing approaches for artist identification.
   •	Explore and visualize the learned feature representation for identifying artists.
   
7. This notebook will purely be an exploratory and hopefully concise enough attempt to explain the idea as well as using different methods to extract meaningful relations out of it.

8. My stakeholders (Art lovers, Art industries, Computer Vision works working with Arts, Museums and government) will be greatly benefitted with my model and also common people can predict artist by looking at arts.

# STEP-2 Data acquisition:

1. we first obtain a large dataset of art compiled by Kaggle that is based on the WikiArt dataset [13]. This dataset contains roughly 100,000 paintings by 2,300 artists spanning a variety of time periods and styles. The images vary widely in size and shape. 

2. Every image is labeled with its artist in a separate .csv file. The vast majority of artists in the full dataset have fewer than 50 paintings, so in order to ensure sufficient sample sizes for training networks we use only the artists with 300 or more paintings in the dataset. Therefore, our dataset consists of 300 paintings per artist from 57 artists (about 17,000 total paintings) from a wide variety of styles and time periods.

# STEP-3 Data Cleaning:

1. Because the art in the dataset comes in a variety of shapes and sizes, we modify the images before passing them into our CNNs. First, we zero-center the images and normalize them. 

2. Next, we take a 224x224 crop of each input image. While training the network, we randomly horizontally flip the input image with a 50% probability and then take a crop of a random section of the painting. This randomness adds variety to the training data and helps avoid overfit. 

3. For the validation and test images, we always take a 224x224 center crop of the image to ensure stable and reproducible results. We do not rescale paintings before taking crops in order to preserve their pixel-level details. Our hypothesis is that artist style is present everywhere in an image and not limited to specific areas, so crops of paintings should still contain enough information for a CNN to determine style.

4. Also, we hypothesize that in order to determine style, it is important to preserve the minute details that might be lost with rescaling. Given the large number of entries in the dataset and the processing that is needed before passing them into our CNNs, we do not load our entire dataset into memory but store it on disk and load minibatches one at a time. This slows down training due to requiring additional disk read time, but allows us to train using our entire dataset and to use larger crops of our paintings than would be possible otherwise, improving overall accuracy.

5. We develop and train three different CNN architectures to identify artists. Every network we use takes as input a 3x224x224 RGB image and outputs the scores for each of the 57 artists present in our dataset.

6. For all of our networks, we use a softmax classifier with cross-entropy loss.
This loss function ensures that our network is constantly trying to maximize the score of the correct artists of its training examples relative to other artists during training.

# STEP-4 Feature Selection:
# STEP-5 Modeling:
# STEP-6 Evaluation:
# STEP-7 Conclusion:

#####################################################

Trained different classification models including SVM, RANDOM FOREST, ADABOOST etc and predicts artists from different art batches per artist(2/3/10/50 artist arts).

Artist_Identification_from_Arts_2Artists.ipynb

Artist_Identification_from_Arts-3Artists.ipynb

Artist_Identification_from_Arts-10Artists.ipynb

Artist_Identification_from_Arts-50Artists.ipynb

Base CNN to Resnet-18 Transfer learning:
TrainArtists-Subset.ipynb

Classifying Pablo Picasso Arts from other Artists with 74% accuracy:
Classifying Pablo Picasso Arts from other Artists.ipynb

#####################################################

