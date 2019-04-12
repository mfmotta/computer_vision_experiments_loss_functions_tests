# A new loss function: Yukawa Potential Loss

In *Dimensionality Reduction by Learning an Invariant Mapping* by Hadsell, Chopra, and LeCun, the authors
 propose a novel method for learning a "globally coherent non-linear function that maps the data evenly to the output manifold" (sic).  
 
 This method has two main ingredients: 1) Contrastive loss and 2) Siamese networks (+ pair mining)
 
 Considering images X_i with labels y_i, the data is mined in pairs such that a pair (X_1,X_2) is fed to the Siamese networks, which are just two CNN's with identical parameters. To each pair (X_1,X_2) corresponds a label Y, where Y = 1 if y_1 = y_2 (positive pair) and Y = 0 if y_1 != y_2 (negative pair).
 The Siamese networks output vector encodings G(X_1) and G(X_2). The training aims to minimize the, so called, Contrastive Loss. The Contrastive loss is inspired by the dynamics of a mechanical spring system, for which there is a repulsive force for distances smaller than the equilibrium distance, and an attractive force, otherwise. The spring which corresponds to the contrastive loss has the unnatural property of having equilibrium distance = 0 for equally labeled (positive) pairs  and breaking down for distances bigger than some threshold 'm'. See details in the original paper.
 
 Here, we make experiments modifying the main ingredients 1) and 2) above. 
 
 First, I change 1) and test a different loss function, which I call Yukawa loss, or Potential loss, because it is based on the [Yukawa potential](https://en.wikipedia.org/wiki/Yukawa_potential). Just as Hadsell et al have used the [spring model] (https://en.wikipedia.org/wiki/Hooke%27s_law) in order to model the 'attraction' and 'repulsion' between positive and negative pairs respectively, here I use the Yukawa potential to model the repulsion between negative pairs and also a cubic function for equally labeled pairs. 
 
 ### String Potential: Contrastive Loss: 
 L = L_pos + L_neg = Y*D² + (1-Y)*(m-D) , replacing the values of Y for positive and negative pairs:
 
 L_pos = D² 
 
 L_neg = m-D
 
 <p align="center">
 <img src="https://github.com/mfmotta/siamese_keras/blob/master/images/string_potential.png"  width=300">
 </p>
 
### Yukawa vs.Contrastive Loss 
 L = L_pos + L_neg = Y*D³ + (1-Y)*Exp(-10*(1-Y)*D)/D, replacing the values of Y for positive and negative pairs:
 
 L_pos = D³ 
 
 L_neg = Exp(-10*D)/D

 <p align="center">
 <img src="https://github.com/mfmotta/siamese_keras/blob/master/images/yukawa_cubic.png"  width=300">
 </p>

Notice that the proposed loss function is steeper for both positive and negative pairs distances and has a lower minimum. Let us see what results from these changes.


## Results:
We consider the following simple MLP:

input -> Flatten -> Dense (128) + Dropout (0.1) -> Dense (128) + Dropout(0.1) -> Dense(128)

The weights are initialized with the same values for training with each loss function. 
  
### 20 epochs - batch size = 128

After 20 epochs we see that the Potential loss has slightly better performance on the test set, scoring 98.10% accuracy against 97.37% with the Contrastive loss. It is interesting to see the distances between the embedding vectors after training. Here we show the distributions of distances between positive and negative pairs on the test set:
<p align="center">
<img src="https://github.com/mfmotta/computer_vision_experiments_loss_functions/blob/master/contrastive_vs_yukawa/images/dist_positive_pairs_20epcs_128.png"  width="350"> <img src="https://github.com/mfmotta/computer_vision_experiments_loss_functions/blob/master/contrastive_vs_yukawa/images/dist_negative_pairs_20epcs_128.png"  width="350">
  </p>
We can see that the Contrastive loss does a better job at clustering the positive pairs together, with mean distance = 0.030 and standard deviation = 0.046, whereas the Potential loss has mean dist=0.052 and std=0,058. For negative pairs, the Potential loss is able to better cluster negative pairs together, with mean dist = 0.795, std=0.104, while these values for the Contrastive loss are mean dist= 1.171 and std=0.199.

(Ideally, we should have repeated the training with settings above n times with different weights initialization and shuffling the samples, sut since this is not a paper, I'm only informally reporting results after some repetitions.) 

We can also visualize the t-SNE projections of the test set

Pre training:
<p align="center">
 <img src="https://github.com/mfmotta/computer_vision_experiments_loss_functions/blob/master/contrastive_vs_yukawa/images/t_SNE_pre-training.png"  width="500">
 </p>
  
 After training with the Contrastive loss:
 
 <p align="center">
 <img src="https://github.com/mfmotta/computer_vision_experiments_loss_functions/blob/master/contrastive_vs_yukawa/images/t_SNE_20epcs_128_contrastive.png"  width="500">
 </p>
 
  After training with the Yukawa (potential) loss:
 
  <p align="center">
 <img src="https://github.com/mfmotta/computer_vision_experiments_loss_functions/blob/master/contrastive_vs_yukawa/images/t_SNE_20epcs_128_potential.png"  width="500">
 </p>

### 40 epochs - batch size = 128

<p align="center">
<img src="https://github.com/mfmotta/computer_vision_experiments_loss_functions/blob/master/contrastive_vs_yukawa/images/dist_positive_pairs_40epcs_128.png"  width="350"> <img src="https://github.com/mfmotta/computer_vision_experiments_loss_functions/blob/master/contrastive_vs_yukawa/images/dist_negative_pairs_40epcs_128.png"  width="350">
  </p>


 Contrastive loss:
 
 <p align="center">
 <img src="https://github.com/mfmotta/computer_vision_experiments_loss_functions/blob/master/contrastive_vs_yukawa/images/tSNE__40epcs_128contrastive.png"  width="500">
 </p>
 
 Yukawa loss:
 
  <p align="center">
 <img src="https://github.com/mfmotta/computer_vision_experiments_loss_functions/blob/master/contrastive_vs_yukawa/images/tSNE__40epcs_128potential.png"  width="500">
 </p>
 
 
 
