# VitalSign_Estimator
From multispectral iPPG data on a human face region through a hyperspectral camera, oxygen saturation is estimated accurately and robustly to individual factors. 
* Methods: For estimating oxygen saturation, iPPG data of four ROI areas: forehead, under the eyes, cheeks, and under the nose of a person's face are used as input.  A deep learning model for estimating melanin and skin thickness corresponding to individual factors is constructed, and oxygen saturation is estimated by using the estimated value and iPPG data as input to the stacked LSTM-based model. Finally, the estimation accuracy is improved by fusing the estimated values for the four ROI regions through CPF. 
* Results: The validation was performed on 12 healthy individuals, and when melanin and skin thickness for estimating oxygen saturation were used together as inputs, the accuracy was higher than when not. The oxygen saturation estimation result through the proposed model showed a low error rate even in the abnormal situation where the oxygen saturation value fell, and the overall error rate did not exceed 2%. Additionally, the error was reduced as a result of fusion of the estimation results of each ROI through CPF. Conclusion: Through the proposed model, it is possible to estimate the oxygen saturation robustly and accurately depending on individual factors. Moreover, it is possible to accurately estimate the degree of dispersibility even in an abnormal situation in which the oxygen saturation level drops or changes significantly. 

# Model for Estimating Melanin and Thickness 
 The multi-spectral image of the skin includes information on absorption by the chromophore of arterial blood vessels as well as information on absorption by skin tissue. The parameters that indicate absorption by the skin tissue are melanin and skin thickness, and since they vary depending on the person, they may have an effect as an individual error factor when estimating arterial blood oxygen saturation through multi-spectral images. For this reason, it is necessary to estimate melanin and skin thickness information from multi-spectral images for accurate oxygen saturation estimation, and then add that information. <br>
 Most of the skin parameter extraction studies for multi-spectral images based on artificial neural networks were performed using a regression method based on a model composed of a simple fully connected layer. Therefore, in this paper, for estimating melanin and skin thickness, a probability-based regression model consisting of three major models is proposed: an embedding model, a classification model, and a regression model as shown in Figure.<br><br>
<img width='80%' height='80%' src='https://user-images.githubusercontent.com/60689555/233256045-2673696d-9e0e-4f4f-bcb8-c0d6af6ba08d.png'><br><br>
 In order to estimate melanin and skin thickness based on probability, the melanin and skin thickness values should be classed first. Since melanin generally has a value of 0 to 15%, it was divided into a total of 8 classes with a unit of 2%. Moreover, since the skin thickness of the human face area is generally 30um ~ 65um, the skin thickness value was divided into three classes: smaller than 35um, 35um~55um, and larger than 55um. After that, to improve the performance in the classification process, the input absorbance data is clustered so that the same class data has a similar value and is distinguished from other classes through an embedding model. The embedding model was composed of 5 fully connected layers, and Leaky Relu was used as the activation function. The output of the model was clustered by class using triplet loss as a loss function for training.<br>
 
 # Model for Estimating Oxygen Saturation 
 In general, the absorption data for estimating oxygen saturation has time series characteristics. That is, the characteristics of how absorbance changes over time have significance in estimating oxygen saturation. Therefore, a network of stacked LSTM structures is proposed to reflect this time series’ properties and to express the complex relationship with oxygen saturation such as shown in Figure. Stacked LSTM is a structure in which LSTM cells are interconnected with adjacent cells and stacked in multiple layers, and the output of the previous LSTM layer is used as the input of the next LSTM layer. <br><br>
 <img width='80%' height='80%' src='https://user-images.githubusercontent.com/60689555/233257568-54ca528a-cd30-4f75-a1ce-49317743543d.png'><br><br>
 The input $u_t$ for the LSTM cell of the model uses the measured absorbance and the estimated melanin and skin thickness information through the model proposed, which allows for a more robust estimation of oxygen saturation based on personal factors. At this time, the measured absorbance dimension is 14 while the dimension of the regression model estimation value of melanin and skin thickness is relatively as small as 1, so the effect of melanin and skin thickness values may not be efficiently reflected. <br>
Therefore, when estimating oxygen saturation, the value used together as an input was composed to use the probability distribution by class, which is the result of the classification model, rather than the estimation result of the regression model for melanin and skin thickness. In other words, the input of LSTM cells is a vector of the form where the measured absorbance $x_1, x_2, …, x_14$ and the estimated probability distribution of melanin $p_{m1},p_{m2}, …, p_{m8}$ and the probability distribution $p_{t1}, p_{t2}$ and $p_{t3}$ of the skin thickness are concatenated.
Also, when the length of the input sequence of the network is k, the input of the LSTM of the first layer for estimating oxygen saturation at time step $t$ utilizes the vector $u_{t−k−1:t−1} from $t−k−1$ time to $t−1$ time.<br>
The input vector is outputted to the oxygen saturation of the final t time step after the Stacked LSTM and two-layered Fully Connected Network. At this stage, as the sequence data used as input, the measured data is transformed to 30 frame per sec (fps) using linear interpolation, and the length of the sequence is set to 100. Moreover, the number of hidden nodes between each LSTM cell is set to 30. After that, the output values of all nodes of the second LSTM layer are used as inputs of the Fully Connected layer. Additionally, RMSE was used as a loss function for model training, and Adam Optimizer was used as the optimizer.
