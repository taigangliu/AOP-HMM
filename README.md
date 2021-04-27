AOP-HMM
=========================
Antioxidant proteins (AOPs) play important roles in the management and prevention of several human diseases due to their ability to neutralize excess free radicals. However, the identification of AOPs by using wet-lab experimental techniques is often time-consuming and expensive. In this study, we proposed an accurate computational model, called AOP-HMM, to predict AOPs by ex-tracting discriminatory evolutionary features from hidden Markov model (HMM) profiles. First, auto cross covariance variables were applied to transform the HMM profiles into fixed-length feature vectors. Then, we performed the analysis of variance (ANOVA) method to reduce the di-mensionality of the raw feature space. Finally, a support vector machine (SVM) classifier was adopted to conduct the prediction of AOPs. To comprehensively evaluate the performance of the proposed AOP-HMM model, the 10-fold cross validation (CV), the jackknife CV and the inde-pendent test were carried out on two widely used benchmark datasets. The experimental results demonstrated that AOP-HMM outperformed most of the existing methods and could be used to quickly annotate AOPs and guide the experimental process.


Installation Process
=========================
Required Python Packages:

Install: python (version >= 3.5)  
Install: sklearn (version >= 0.21.3)  
Install: numpy (version >= 1.17.4)  
Install: PyWavelets (version >= 1.1.1)  
Install: scipy (version >= 1.3.2)  

pip install < package name >  
example: pip install sklearn 

Usage
=========================
To run: $ ACC_ANOVA_SMOTE.py or ACC_ANOVA.py
