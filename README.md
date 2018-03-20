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

1. Ran test in Google Cloud Datalab using GPUs to handle large features.

# STEP-5 Modeling:

1. Trained different classification models including SVM, RANDOM FOREST, ADABOOST, Logictic Regression etc.
2. Trained Artificial Neural Network.
3. Trained Convolution Neural Network(CNN) after data is properly augmented.

# STEP-6 Evaluation:

1. Used classification accuracy, precision/recall/F1 score/ confusion metrics as evaluation metrics.

Using the scores generated by our networks, we report top-1 classification accuracy (the fraction of paintings whose artists are identified correctly), precision, recall, and
F1 scores. We also report top-3 classification accuracy, which considers a painting correctly classified if the correct artist is in the top 3 highest scores generated by a network. 

# STEP-7 Conclusion and Future Work:

1. We introduce the problem of artist identification and apply a variety of CNN architectures to it to maximize clas- sification accuracy, which has not been done in prior work. Our dataset consists of 300 paintings per artist for 57 artists over a wide variety of styles and time periods.

2. For future work, we would like to dive deeper into the model representations and try to quantify how much of the predictions come from the style of an image versus the content. Our belief is that the networks create a representation primarily of the style of a artists, but there are likely some elements of content present as well - for example, Mary Cassatt almost exclusively painted young women, so the network might have learned a representation of the content of her paintings to help classify them. 

3. In order to do this, we plan to calculate the Gram matrices of various layers of our network (to use as a representation of style) and pass them into a separate CNN to obtain artist predictions. This network would still produce one score per artist, but instead of looking at the entire image as an input, it only looks at the information representing image style. We would then compare predictions from this new neural network with our best-performing network to understand how much of the artist predictions comes from the style versus other aspects of a painting like the content.

4. We would also like to expand our dataset and see how our network handles classifying more artists with fewer paint- ings for artist. We used 57 artists with more than 300 paint- ings, but our original dataset has 108 artists with more than 200 paintings. We would switch to using all available images for the artists we use in our dataset instead of using an equal number per artist. This would result in an unbalanced dataset, but if we expand to using more artists, we should still be able to take advantage of the larger sample sizes for certain artists and classify them with high accuracy.

## Jupyter Notebooks for different experiements.##

Trained different classification models including SVM, RANDOM FOREST, ADABOOST etc and predicts artists from different art batches per artist(2/3/10/50 artist arts).

# Artist_Identification_from_Arts_2Artists.ipynb
       --Sample 1000 paintings from 2 artist(500 arts each) --84% classification accuracy, 84% precision/recall.

# Artist_Identification_from_Arts-3Artists.ipynb
       --Sample 1500 paintings from 3 artists(500 arts each) --79% classification accuracy, 79% precision/recall.
       
# Artist_Identification_from_Arts-10Artists.ipynb
       --Sample 5000 paintings from 10 artists(500 arts each) --49% classification accuracy, 49% precision/recall.
       
# Artist_Identification_from_Arts-50Artists.ipynb
       --Sample 22629 paintings from 50 artists --34% classification accuracy, 34% precision/recall.

# Artist Identification with Convolutional Neural Networks 
       --Sample 1000 paintings for 3 artists CNN model.
       
# TrainArtists-Subset.ipynb
       --Base CNN to Resnet-18 Transfer learning.

# Classifying Pablo Picasso Arts from other Artists-Final.ipynb
       --Classifying Pablo Picasso Arts from other Artists with 74% classification accuracy.

