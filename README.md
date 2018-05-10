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

# 5.1. Baseline CNN
We train a simple CNN from scratch for artist identifi- cation. Table 1 shows the architecture of this network. As the name implies, this network serves as a baseline for com- parison with the other approaches. Every layer in the net- work downsamples the image by a factor of two in order to reduce computational complexity, but the downside of this approach is that it might not allow sufficient exploration of lower-level features as the image details are quickly aggre- gated.

Input size
Layer
3x224x224
3x3 CONV, stride 2, padding 1
32x112x112
2x2 Maxpool
32x56x56
3x3 CONV, stride 2, padding 1
32x28x28
2x2 Maxpool
1x6272
Fully-connected
1x228
Fully-connected
Table 1. Baseline CNN architecture (ReLU and batch normaliza- tion layers omitted for readability).

# 5.2. ResNet-18 Trained from Scratch
Our next network is based on the ResNet-18 architec- ture but with a new fully-connected layer to allow for artist predictions. ResNets use residual blocks to ensure that up- stream gradients are propagated to lower network layers, aiding in optimization convergence in deep networks [7]. We train this network from scratch to allow the network to learn features solely for the purpose of artist identification. Our network architecture is visible in Figure 3. We used the 18-layer version of ResNet in order to allow for faster training and to reduce the memory requirements.

# 5.3. ResNet-18 with Transfer Learning
Our final network is also based on ResNet-18 but starts with pre-trained weights from the ImageNet dataset. Like the previous network, we replace the final fully-connected layer with a new one to calculate a score for each artist in our dataset instead of a score for ImageNet classes.
We start with a pre-trained network to test whether or not a feature representation from ImageNet is a valuable start- ing point for artist identification. Some artists, for exam- ple Renaissance painters, used shapes and objects that you would expect to find in ImageNet since they usually painted lifelike scenes. However, other artists such as Cubists did not paint scenes as directly representative of the real world.


# STEP-6 Evaluation:

1. Used classification accuracy, precision/recall/F1 score/ confusion metrics as evaluation metrics.

Using the scores generated by our networks, we report top-1 classification accuracy (the fraction of paintings whose artists are identified correctly), precision, recall, and
F1 scores. We also report top-3 classification accuracy, which considers a painting correctly classified if the correct artist is in the top 3 highest scores generated by a network. 

# 6.1. Implementation Details
We trained all of our models using an Adam update rule [15]. We explored using SGD with momentum, but obtained better results with Adam. For the two networks trained from scratch, we started with the default Adam pa- rameters of learning rate = 10−3, β1 = 0.9, and β2 = 0.999. We observed the accuracy and loss for both the train- ing and validation datasets over the training epochs and de- creased the learning rate by a factor of 10 if improvement slows down significantly. We initialized the weights of the convolutional layers of our networks based on [8], as the methodology from this work is best practice when initializ- ing networks that use a recitified linear activation function.
For training the ResNet with transfer learning, we first held the weights of the base ResNet constant and updated only the fully-connected layer for a few epochs. We per- formed this step using the same default Adam parameters described previously. After network performance stopped improving, we allowed weights throughout the entire net- work to change but lower the learning rate to 10−4. This allows for some change throughout the entire network to better fit our dataset.
We experimented with varying levels of L2 regulariza- tion on all networks but did not see significant changes on validation set performance so in the end we used no regu- larization during training.

# 6.2. Evaluation Metrics
Using the scores generated by our networks, we re- port top-1 classification accuracy (the fraction of paintings whose artists are identified correctly), precision, recall, and
F1 scores. We also report top-3 classification accuracy, which considers a painting correctly classified if the cor- rect artist is in the top 3 highest scores generated by a net- work. We compare our networks against each other as well as against [26], which reports SVM classification accuracy using pre-defined features.
Precision and recall are defined as:
P recision = T rueP ositives
T rueP ositives + F alseP ositives
T rueP ositives
Recall = T rueP ositives + F alseN egatives
The F1 score, a weighted average of precision and recall, is defined as:
F1 = 2 ∗ Precision ∗ Recall P recision + Recall
In addition to quantifying the accuracy of our networks, we conduct a variety of experiments to qualitatively evalu- ate their performance and to understand their learned rep- resentations. We use a variety of techniques including saliency maps, gradient ascent for class score maximiza- tion, and filter visualizations [18].

# 6.3. Quantitative Analysis

We can see that both networks that were trained from scratch do not perform as well as the SVM approach from prior work and that the network that started out pre-trained on ImageNet performs signif- icantly better, with a 20% greater absolute (not relative) Top-1 training accuracy. When looking at Top-3 classifi- cation accuracy, we see that all networks perform better than their Top-1 accuracies. Our best network achieves a Top-3 classification accuracy of almost 90%, meaning that it can narrow down the artist to three in the vast majority of cases. Of the networks trained from scratch, ResNet-18 outperforms the baseline CNN by a large margin, indicating that increasing network depth does help improve accuracy. Network performance as ranked by test set accuracy is con- sistent with performance on the other reported metrics as well. Although there is variation in the F1, precision, and recall scores, they tend to track closely with classification accuracy.

We see that in the networks trained from scratch, training and validation accuracies track very closely together, indicating little overfit. This could mean that performance of these net- works would improve with more training epochs and more training data as we generally expect to see a small but no- ticeable gap between training and validation/test accuracy in a fully-trained network. The pre-trained network does have a gap between training and test accuracy, indicating that adding more training data and further training might not add as much value.

Despite many hyperparameter tweaks as learning slowed over time, the baseline CNN improved relatively slowly and peaked at round 40% classification accuracy. ResNet-18 when trained from scratch however, shows clear and definite improvements throughout the training process. Each time that learning slowed and the learning rate was decreased by a factor of 10, we saw a clear and immediate bump in accu- racy. This happened multiple times, which is why there are multiple plateaus and upward jumps following them in the accuracy graphs. In the ResNet-18 with transfer learning, we see one notable jump after epoch 5, and a smaller jump at epoch 11. The first jump at epoch 5 corresponds to not only a learning rate decrease but also to allowing the entire network to change instead of only the final fully-connected layer. We see a significant increase in training and valida- tion accuracy at this pint, indicating that the network be- gins adapting much better to an art dataset. 

One artist in the middle, Henri Matisse, is light blue; only 11 of his 30 paintings in the test dataset were correctly attributed to him. It does not appear that Matisse is being strongly confused for any other artist in particular, as there are not other artists in his row with bright colors. Instead, his paintings have been attributed to a wide variety of artists. Matisse is most confused for Martiros Saryan; the network predicted Saryan as the artist for 3 of Ma- tisse’s paintings. This is not an entirely unexpected result, as Saryan’s style was heavily influenced by Matisse [4]. Ad- ditionally, Matisse experimented and painted in a wide vari- ety of styles, ranging from Fauvism to Post-Impressionsim to Modernism, increasing his overlap with other artists [21]. These results point towards our network building a repre- sentation of artistic style and having trouble if an artist has a wide variety of styles and influenced others.

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

