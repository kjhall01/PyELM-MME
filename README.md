Methodologies:
	- Multiple Linear Regression (MLR)
		> Traditional Multiple Linear Regression
	- Principle Components Regression (PCR)
		> Multiple Linear Regression of data transformed to orthogonal space
		> Eliminates Multicollinearity problems in highly correlated data features
	- MLR using Singular Value Decomposition solving algorithm (SVD)
		> MLR-SVD for alternate implementation
		> SVD also addresses multicollinearity in input data features
	- Traditional Ensemble Mean (EM)
		> Mean of input data features
	- Bias Corrected Ensemble Mean (BCEM)
		> Mean of input data features with bias correction applied
	- Extreme Learning Machine (ELM)
		> ELM algorithm for training artificial neural networks
	- ELM with PCA (ELM-PCA)
		> ELM algorithm applied to data transformed to orthogonal space by PCA
		> PCA Eliminates Multicollinearity problems in highly correlated data features
	- PCA-ELM (PCA-ELM)
		> ELM algorithm applied to data transformed to orthogonal space by PCA
		> PCA-calculated Eigenvectors used to initialize weights of artificial neural network (W)
		> Number of hidden neurons in neural network equal to number of principle components needed
			to retain X% of variability in input data
		> Bias vector for Neural network set to constant to eliminate random variation
		> Method optimizes time to convergence of neural network weight
	- Single Layer Feed-forward Neural Network (SLFN)
		> Traditional stochastic gradient descent approach to artificial neural network
