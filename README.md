# VISTA ( Visual Identification of Significant Travel Attractions)
Automated Photo Filtering for Tourism Domain Using Deep and Active Learning: The Case of Israeli and Worldwide Cities on Instagram

## Abstract:
Social media platforms like Instagram significantly influence tourists' travel decisions by providing them with valuable insights, recommendations, authentic information, and points of interest. 
However, photos shared with location-specific hashtags, even those related to tourist attractions, do not always reflect the actual destination, creating challenges for potential visitors seeking accurate information. 
To assist tourists in finding pertinent tourism information for specific destinations, we propose VISTA: Visual Identification of Significant Travel Attractions. 
The proposed method employs deep learning and active learning techniques to automatically classify photos into: 'Tourism-Related' photos (i.e., photos related to tourism) and 'Non-Tourism-Related' photos (i.e., photos unrelated to tourism). 
To train our machine learning classifier, we created a dataset containing photos of the 10 most popular Israeli cities on Instagram. The classifier obtained an accuracy score of 0.965 and a weighted F1 score of 0.964. 
Evaluating our classifier's global generalization on the InstaCities100K dataset, derived from InstaCities1M, yielded an accuracy score of 0.958 and a weighted F1 score of 0.959. 
The effectiveness of VISTA was demonstrated by comparing tourism-related and non-tourism-related photos in terms of photo proportion, user engagement, and object comparison. We found that most photos published on Instagram associated with cities are irrelevant to tourists and that tourism-related photos received more likes than non-tourism-related photos. 
Finally, there was a low overlap between objects in the two photo collections. 
Based on these results, we conclude that VISTA can help tourists tackle the problem of finding relevant tourism-related photos among the high volume of photos available on Instagram.

## files:
1. Paper_Visualizations.ipynb -> A Jupyter notebook for generating our figures and graphs.
2. predict_image_by_classifier.py -> A toy sample to predict using the VISTA classifier on a photo.  
3. train_test_classifier.py -> A Python file used to train and predict the VISTA classifier.
4. test_set_generation_InstaCities100K_dataset.py -> A Python file used to generate the test set based on the InstaCities100K dataset.
5. classifier_performance.py -> A Python file used to test the VISTA classifier.
6. predict_classiifer_test_InstaCities100K_dataset.py -> A Python file used to predict on the InstaCities100K dataset using the trained VISTA classifier.

## BibTeX Citation
If you use VISTA in a scientific publication, we would appreciate using the following citation:

```
@article{paradise2024,
    doi       = {},
    url       = {},
    year      = {2024},
    publisher = {},
    volume    = {},
    number    = {},
    pages     = {},
    author    = {},
    title     = {},
    journal   = {}
}

