🔌 Energy Predictor LSTM

An end-to-end deep learning solution for forecasting hourly energy consumption in a smart building using LSTM (Long Short-Term Memory) networks.

📌 Project Overview
This project presents a time-series forecasting pipeline that leverages LSTM neural networks to predict the energy usage of a smart home. The model is trained on environmental sensor data (e.g., temperature, humidity) and targets energy consumption in appliances, offering a foundation for energy optimization and load forecasting.



🧠 Technologies Used

- Python
- Pandas, NumPy (Data manipulation)
- Scikit-learn (Preprocessing)
- TensorFlow / Keras (Deep learning-LSTM)
- Matplotlib / Seaborn (Data Visualization)

 🔄 Pipeline


1. Data Ingestion
   Load the energy dataset and handle missing values or anomalies.

2. Feature Engineering & Normalization
   Select relevant features (temperature, humidity, etc.) and normalize values using `MinMaxScaler`.

3. Windowed Sequence Generation
   Create overlapping time windows to feed into the LSTM model (sequence-to-one approach).

4. Model Training
   Train an LSTM neural network with early stopping and validation tracking.

5. Evaluation & Visualization
   Plot predictions vs actuals and visualize training vs validation loss.

6. Model Saving 
   Export the trained model in `.h5` format for potential real-time deployment. 

 📊 Dataset

UCI Appliances Energy Prediction Dataset  
[Download link](https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv)

 🧩 Business Impact


- ⚙️ Smart Home Automation: Enables edge deployment for real-time energy regulation and cost savings in smart homes.
- 🏭 Industrial Applications: Supports predictive maintenance and consumption forecasting in energy-intensive facilities.
- 🔌 Scalable AI Infrastructure: Aligns with Eksim Holding’s vision for AI-driven energy management and sustainability solutions.





---

🔍 Model Prediction Result

 🔍 Model Prediction Result

The plot below compares the predicted and actual hourly energy consumption over the first 100 hours of the dataset.

📊 Evaluation Metric: Mean Squared Error (MSE) was used to evaluate model performance.

This output demonstrates that the LSTM model successfully captures time-dependent patterns in energy usage.  
Such forecasting is essential for:

- ⚡ Smart grid management  
- ⏱ Real-time energy optimization  
- 💸 Reducing operational costs through predictive insights

![Tahmin Görseli](prediction_plot.png)

---

📉 Training Loss
This plot illustrates how the training loss decreased over time during model training.
It reflects the model’s learning progress and shows that the LSTM architecture successfully adapted to the energy consumption patterns in the dataset.

![Eğitim Loss Grafiği](training_loss_plot.png)

📉 Validation vs Training Loss

This plot compares the training and validation loss values across epochs.
It helps assess the model’s ability to generalize by visualizing the difference between how well it learns from training data and how well it performs on unseen data.
A small gap between the curves indicates a well-generalized model with minimal overfitting.

![Validation vs Training Loss](val_vs_train_loss.png)

