# YouTube Dislikes Dataset / Prediction

YouTube dislike prediction using Python, YouTube Data API v3, TensorFlow/Keras.

## Project Structure

The project has the following structure:

```

```

The `datasets` folder contains the data and notebooks for getting and cleaning the data:
- `youtube_kaggle_dataset` - dataset from Kaggle:
  - The `data` folder contains [YouTube Trending Video Dataset downloaded from Kaggle](https://www.kaggle.com/rsrishav/youtube-trending-video-dataset) and clean version of it.
  - The `data_cleaning_kaggle.ipynb` notebook contains code to clean Kaggle dataset.
  - The `parse_json_categories.ipynb` notebook contains code to convert JSON categories file to python dictionary.
 
- `youtube_custom_dataset` - custom collected dataset:
  - The `api_keys` folder contains text files with YouTube API keys. To know more, visit [official documentation]().
  - The `data` folder contains data collected using YouTube Data API. To download dataset, visit [YouTube Dislikes Dataset on Kaggle]().
  - The `video_IDs` folder contains text files with unique video IDs.
  - The `youtube_API_requests_examples.ipynb` notebook contains examples of YouTube Data API v3 requests to get information about videos and comments to explore YouTube responses structure.
  - The `youtube_video_id.ipynb` notebook contains code to get unique video IDs from kaggle dataset and/or to generate random ids.
  - The `dataset_collection.ipynb` notebook contains code to collect YouTube Dislikes Dataset.