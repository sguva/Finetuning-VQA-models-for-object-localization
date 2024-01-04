## Finetuning Visual Question Answering (VQA) models for object localization which can help visually impaired people
This is part of final project of Natural Language Processing course.
For complete code of the project please check: https://github.com/sguva/NLP-Project


### Instructions:
Before running any notebooks, create a python environment and activate it.

```bash
conda create -n vqa_loc python=3.10 -y

conda activate vqa_loc

pip3 install torch torchvision torchaudio
pip3 install -r requirements_pip.txt
```

Check the **instructions_executing_files.ipynb** notebook for data generation, training, and validation.

Check classification_demo.ipynb and answer_generation_demo.ipynb for inference demo.

### Summary:
Navigating the world becomes a constant challenge for visually impaired individuals due to their difficulty in pinpointing the location of objects around them. This limitation hinders their ability to interact with their surroundings and hampers their independence. There's onging research at the intersection of natural language processing and computer vision with existing models, like VQA and image captioning, are adept at answering questions about object identity, attributes, and relationships. However, they lack the capability to pinpoint locations, as their training datasets rarely include annotations about spatial positioning. This leaves visually impaired individuals relying on incomplete or irrelevant information, such as "the cat is in front of the dog," when seeking specific object locations in the image. Hence finetuning a VQA model specifically to answer location-based questions about images may help with this problem as the Vision Transformers divide images into patches and encode them using self-attention mechanisms. These mechanisms allow for global interactions between patches, potentially capturing long-range dependencies and relationships between objects. This problem can be sort of achieved using object detection networks with additional pre and post processing but it doesn't involve understanding text and we can't go beyond the 80 object categories of the COCO dataset.

Approach 1: Finetuning a VQA model for location classification task.

Approach 2: Finetuning a VQA model to generate text with location information.

Data generation: Divide the input image into nine regions, top-left, top, top-right, left, center, right, bottom-left, bottom, bottom-right. This is only for identifying the target region and the input to the network will be the complete image. Then for each instance of an object get the information about which of the nine regions the object is located based on the center of the bounding box. For the classification task these nine regions are the nine classes.


The code is based on the following sources:
1. COCO data generation: https://github.com/ShubhankarPoundrik/NLP_Project
2. Finetuning for classification task: https://huggingface.co/docs/transformers/main/tasks/visual_question_answering
3. Finetuning for text generation: https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BLIP-2/README.md

References:
1. COCO Dataset: https://cocodataset.org/
2. BLIP-2: https://arxiv.org/abs/2301.12597 & https://huggingface.co/docs/transformers/main/model_doc/blip-2
3. ViLT: https://arxiv.org/abs/2102.03334 & https://huggingface.co/docs/transformers/model_doc/vilt


### Limitations & Improvements:

1. Bias in image datasets and weighted loss:
One aspect in general about image based datasets is that most of images contain objects in the center as most of the photos are intended to be captured that way and we found out that it is the same scenario for the COCO dataset, and that the model will be biased towards towards the middleCenter location. The current implementation uses the default loss functions provided by the model classes of ViLT and BLIP2 itself but if we use weighted loss function i.e., giving lower weight to classes with more data samples and higher weight to the underrepresented classes of the dataset then it's possible that the model will make even better predictions and won’t be biased to the classes with more number of samples. We investigated and worked on this approach to some extent but ran into issues while implementing it and also due to the huge amount of time and data requirement for these models, we couldn’t complete this enhancement of our presented work.

2. Fine-tuning BLIP2 with Additional data:
After seeing the BLIP2 model’s performance on the validation set it’ll be a good idea to also finetune the BLIP2 model using image captions available in the COCO-dataset as well along with the generated data, in that way the model wouldn’t overfit and would be able to still retain/learn knowledge about the images in general apart from just the location information. As training/finetuning BLIP2 like models requires extensive compute power and huge amount of time.