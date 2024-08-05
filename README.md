<h1>Disentanglement for pain classification using TDGAN architecture</h1>

<h2>Approch</h2>
<p>This work is inspired by the disentanglement approch for facial expression recognition proposed proposed by Xie et al. (2020) </p>
<p>Pain being a subtle expression facing entanglement issues with this model we want to separate pain and identity with this two-branch disentangled generative adversarial network. 
  It manages to disentangle identity and pain by training to generate from two images a new fake one with the identity of the fisrt image and the expression of the second one </p>

  <p><img alt="Image" title="architecture" src="images/architecture.png" /></p>

  XIE, Siyue, HU, Haifeng, et CHEN, Yizhen. Facial expression recognition with two-branch disentangled generative adversarial network. <em>IEEE Transactions on Circuits and Systems for Video Technology</em>, 2020, vol. 31, no 6, p. 2359-2371.


<h2>Database</h2>

<p>We evaluate the performance of our model using the BioVid Heat Pain Database, a dataset designed for pain recognition research. The database is segmented into five distinct parts, each varying in terms of subjects, labeling, modalities, and tasks.
In our experimentation we use Part A of the dataset, which includes data from 87 subjects, encompassing four pain levels in addition to a neutral state. 
The modalities in Part A consist of frontal video recordings and biological signals. In this study, we concentrate on the highest pain level and the neutral state, using only the frames extracted from the frontal video recordings.

Our experimental protocol consists of a 5-folds cross-validation where training set contains 49 subjects, the validation set contains 12 subjects and the test set 21 subjects.

WALTER, Steffen, GRUSS, Sascha, EHLEITER, Hagen, et al. The biovid heat pain database data for the advancement and systematic validation of an automated pain recognition system. <em>IEEE international conference on cybernetics (CYBCO)</em>. IEEE, 2013. p. 128-131.</p>

<h2>Results</h2>

<p>The results are presented as the average accuracy obtained during the testing phase by the five models trained with each of the five folds.
Two different backbones are used. The fisrt one is the Inception style one proposed by Xie et al. and the second one is ResNet-18 (He at al. 2016)

<table>
 <thead>
   <tr>
    <th align="center"></th>
    <th align="center">ResNet-18</th>
    <th align="center">Inception style</th>
 </tr>
</thead>
<tbody>
  <tr>
   <td align="center">Baseline</td>
   <td align="center">58.3%</td>
   <td align="center">59.2%</td>
 </tr>
 <tr>
  <td align="center">Disentanlement with TDGAN</td>
  <td align="center">58.2%</td>
  <td align="center">60.0%</td>
 </tr>

</tbody>
</table>

We notice that an improvement is possible with the disentanglement method compared to the baseline with Inception style backbone.

Here are two examples of pain generation by the model. The fisrt image is the image for which identity is extracted, the second image is the one for which expression is extracted and the third one is the generated one with identity of image 1 and expression of image 2.
<p><img alt="Image" title="architecture" src="images/example1.png" /></p>
<p><img alt="Image" title="architecture" src="images/example2.png" /></p>


HE, Kaiming, ZHANG, Xiangyu, REN, Shaoqing, et al. Deep residual learning for image recognition. In : <em>Proceedings of the IEEE conference on computer vision and pattern recognition.</em> 2016. p. 770-778.

XIE, Siyue, HU, Haifeng, et CHEN, Yizhen. Facial expression recognition with two-branch disentangled generative adversarial network. <em>IEEE Transactions on Circuits and Systems for Video Technology</em>, 2020, vol. 31, no 6, p. 2359-2371.
