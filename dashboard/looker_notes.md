Link to project dashboard: https://lookerstudio.google.com/u/0/reporting/aafa6deb-7c88-48f6-97ae-04e2d1797b93/page/p_uarz070brd

1. Dashboard Overview:
   
      •	Purpose: Monitor fraud detection model performance and transaction patterns
   
      •	Components:
   
            o	Input data analysis: Class distribution, feature correlations.
   
            o	Predictions: Model predictions
   
            o	Output metrics: AUC, F1, precision, recall, accuracy.
   
            o	Confusion matrix: Visualization of confusion matrix and its metrics.
   
2. Input Data Insights:
   
    The V1-V28 features in your dataset are anonymized credit card transaction attributes resulting from Principal Component Analysis (PCA) transformation. While their exact meanings are hidden for privacy or security reasons.
   
    What the data shows:
    
      •	Class Imbalance:
      
          o	🟢 56,855 legitimate transactions (99.8%)
          
          o	🔴 98 fraud cases (0.2%)
          
          o	Features like V10, V12, and Amount show strong patterns. Fraudulent transactions often have negative values in these fields.
          
          o	Transactions with Amount ≈ -0.35 are frequently fraudulent.
   
4. Prediction Accuracy:
   
      •	80.6% of fraud cases (actual=1) were correctly predicted as fraud (true positives).
   
      •	19.4% of fraud cases were missed (false negatives).
    (Actual vs. Predicted) table
    
   ---------------------------Predicted Legit (0)----------Predicted Fraud (1)-------------------------
       
    Actual Legit (0)-----✅ 56,859 (True Negative)-----❌ 5 (False Positive)
    
    Actual Fraud (1)-----❌ 19 (False Negative)--------✅ 79 (True Positive)
    
    The bar chart (0%-100%) shows the ratio of correct vs. incorrect predictions for each class.

6. Output metrics:
   
      This data provides exact counts for auditing.
      
      Metric-----Score-----What It Means
      
      Accuracy-----99.96%-----Misleading (due to imbalance). The AI is mostly guessing legitimate transactions.
      
      Precision-----94%/-----When the AI flags fraud, it’s 94% likely to be correct.
      
      Recall-----80.6%-----AI catches 80.6% of all fraud cases.
      
      AUC-----0.97-----Excellent at distinguishing fraud (0.5 = random, 1 = perfect).
      
      F1 Score-----0.87-----Balanced precision/recall (94% precision, 80.6% recall)

7. Confusion Matrix Highlights:
   
    •	79 true positives (correct fraud detections)
   
    •	5 false positives (legitimate transactions flagged as fraud)
   
    •	19 false negatives (missed fraud cases)
   
    •	56859 true positives (correctly predicted non-fraud cases)
   
This dashboard helps quantify model performance, identify fraud patterns and guide tuning decisions.
