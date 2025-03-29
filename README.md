# PdM_Predictor

# ------ February 2025 Notes ------ #
# Look into inference time correlation with network size
# Determine which models are best for uploading onto embedded systems.
# This can justify why we are using those specific models because it proves others have as well.


# ------ March 11th NOTES ------ #
# - Gather preliminary results with latest changes.
# - Try making candlestick box plot graphs from the results (using seaborn).
# - Document the algorithms and changes well for the paper as you go.
# - Find predictive maintenance health prognostic method that is simple, state-of-the-art, and easy to implement.


# -------- March 25th Notes -------- #
1. [DONE] Main seed should be for whole dataset (3 partitions, and also used for training model).
2. [DONE] All models trained using a given dataset will use this same main seed.
3. [DONE] Full sized images will now be 1800x1080 so that post-processing ROI's matches the full size.
4. [DONE] Before metrics are calculated (after backpropogation), re-stitch the ROI's into a full (1800x1080) image.


[DONE] Generate new dataset (1800x1080)
[] train new models with new setup
[] test new models with new setup


