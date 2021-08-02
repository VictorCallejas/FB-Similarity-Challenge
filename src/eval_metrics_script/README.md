# Facebook AI Image Similarity Challenge Evaluation Script

We are providing a Python script to help competitors calculate competition metrics for the [Facebook AI Image Similarity Challenge](https://www.drivendata.org/competitions/79/competition-image-similarity-1-dev/). This script is able to evaluate submissions for both the [Matching track](https://www.drivendata.org/competitions/79/competition-image-similarity-1-dev/) and the [Descriptor track](https://www.drivendata.org/competitions/80/competition-image-similarity-2-dev/).

To install the necessary dependencies, run:

```sh
pip install -r requirements.txt
```

The script has a command-line interface (CLI). To see the documentation, run:

```sh
python eval_metrics.py --help
```

Note that this script does not contain any of the input validation on submission files that the competition platform will
do (e.g., validating shape of submission, validating ID values). The script may error or produce incorrect results for invalid submissions. Please review the respective Problem Description pages ([Matching track](https://www.drivendata.org/competitions/79/competition-image-similarity-1-dev/page/376/#submissionformat); [Descriptor track](https://www.drivendata.org/competitions/80/competition-image-similarity-2-dev/page/380/#submissionformat)) to ensure your submission files follow the expected format.
