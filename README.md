# PdM_Predictor

# ------ Three experiments for Journal Paper (2025) ------ #

# 1. (ROI 360x360) Segmentaiton using multi-class targets
# 2. (ROI 360x360) Segmentation using binary targets
# 3. (Full-Size 1920x1080) Segmentation using multi-class targets
# 4. (Full-Size 1920x1080) Segmentation using binary targets

# Look into inference time correlation with network size
# Determine which models are best for uploading onto embedded systems.
# This can justify why we are using those specific models because it proves others have as well.


# ------ March 11th NOTES ------ #
# - change source data to include 60 days regardless of whether all hours were captured.
# - change the ROI cutting to occur inside the models themselves. Make a pre-processing method and a post-processing method inside the model files.
# - instead of bar graphs for testing metrics, use plot box graphs (candlestick thingys). (try using seaborn)

Next Steps:
# 1. [DONE] Change source data to include 60 days regardless of whether all hours were captured.
# 2. [DONE] Change dataset generation to use new seeding method.
# 3. [HAVING DIFFICULTY] Change models to incorporate pre- and post-processing algorithms for ROI's.
#       [March 21st Notes]
#       It seems Tensors carry information from the model into the loss function.
#       Therefore, restitching the ROI's after output messes with the Tensor information making it unable to backpropogate.
#       Might not be able to implement this kind of pre- and post-processing inside the model.
#
# 4. Gather preliminary results with latest changes.
# 5. Try making candlestick box plot graphs from the results (using seaborn).
# 6. Document the algorithms and changes well for the paper as you go.
# 7. Find predictive maintenance health prognostic method that is simple, state-of-the-art, and easy to implement.











