Background Removal from Photos using Deep Learning (DeepLabV3+ Architecture, PyTorch)

Overview

This project demonstrates an automated background removal system using the DeepLabV3+ architecture implemented in PyTorch. The model is designed for precise image segmentation, allowing users to remove backgrounds from images accurately and efficiently. The solution leverages a pretrained ResNet backbone and is fine-tuned to improve generalization and performance on background removal tasks.

Features

	•	DeepLabV3+ Architecture: Utilizes a state-of-the-art image segmentation model for accurate background removal.
	•	Pretrained ResNet Backbone: Enhances feature extraction capabilities by leveraging a pretrained ResNet model.
	•	Data Preprocessing & Augmentation: Employs various techniques like scaling, cropping, and flipping to improve model generalization and robustness.
	•	Optimized with Dice Loss: The model is trained using Dice loss to focus on accurate segmentation and background separation.
	•	High Accuracy: Evaluated using accuracy as a key metric, ensuring reliable background removal.

Some Results on Test Dataset

![girl-model-blonde-slav-157666_0](https://github.com/user-attachments/assets/2fe59c62-a4d4-4c31-9939-9dfcac1e3119)


![girl-bicycle-garden-people-630770_0](https://github.com/user-attachments/assets/4c9b53f6-6a13-4e28-b408-5900dee5e30e)
