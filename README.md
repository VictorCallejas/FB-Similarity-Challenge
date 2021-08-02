# Facebook-AI-Image-Similarity-Challenge


## Data
aws s3 cp s3://drivendata-competition-fb-isc-data/all/query_images/ ./data/images --recursive --no-sign-request
aws s3 cp s3://drivendata-competition-fb-isc-data/all/reference_images/ ./data/images --recursive --no-sign-request
aws s3 cp s3://drivendata-competition-fb-isc-data/all/training_images/ ./data/images --recursive --no-sign-request

To download only the first 1000 (for dev local)
--exclude="*" --include="Q00*"
--exclude="*" --include="R000*"
--exclude="*" --include="T000*"


## Interesting repos

https://github.com/facebookresearch/deit

https://arxiv.org/pdf/2102.05644.pdf

https://github.com/google-research/simclr

