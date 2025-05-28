# Utilizing CNN to Detect Driver Drowsiness in Real Time

Drowsy driving is one of the leading causes of traffic accidents worldwide. Detecting driver fatigue in real time can prevent potentially fatal incidents. By leveraging convolutional neural networks (CNNs), this project aims to build an efficient, real-time system capable of identifying signs of drowsiness based on facial features. This approach can enhance vehicular safety and contribute to intelligent transport systems.

**Reference Research Paper**: [Driver Drowsiness Detection Using CNN](https://www.mdpi.com/2478244)

---

## 🎥 Demo  


https://github.com/user-attachments/assets/5fd16067-5ec6-4669-b8cf-289a85f51d77


---

## 📊 Model Results (on test set of 580 images)

To view all these in detail, you can visit the [`train.ipynb`](https://github.com/baelthebard42/Driver-Drowsiness-Detection/blob/main/train.ipynb) file.

**Accuracy on test set**: **97.3%**

---

### 🔍 Precision, Recall, and F1 Score

Precision measures the proportion of correctly identified positive instances out of all instances predicted as positive, while recall measures the proportion of correctly identified positive instances out of all actual positive instances. F1 score combines both precision and recall into a single metric.

![image](https://github.com/user-attachments/assets/5eb9f3e8-07ea-4a9b-95e5-805ecb2148d1)


---

### 🔀 Confusion Matrix

A confusion matrix is a table that evaluates the performance of a classification model by comparing its predictions against the actual (true) labels. The following image illustrates the confusion matrix of our model on test dataset.

![image](https://github.com/user-attachments/assets/2ef64500-b069-4e6f-82bf-c8c8c9c55cf8)



