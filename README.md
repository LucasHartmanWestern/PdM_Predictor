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


# -------- April 3rd Notes -------- #
1. READ PAPERS about PdM and specifically health prognostics. Find good method for classification.
    - Find quantitative method of defining when heat exchanger maintenance is pertinant. Use this information to inform our models.
    - Looking for numerical thresholds that indicate efficiency or operational capacity.
    - Look for ways of determining which areas of heat exchanger are most critical to operational efficiency (so we can possibly define a weighted matrix).
2. Write the methodology for the actual paper so that we don't forget all the technical details.
3. Try to implement several promising classification/ health prognostic models given the output segmentation masks.
4. Need to include hyperparameter tuning (do this once the entire framework is complete):
    - learning rate
    - scheduler?
    - batch size has to be 1 becasue of VRAM constraints.


# -------- May 22nd Notes -------- #
- Regarding papers on health prognostics:
    -> many approaches are valid, depends on the situation.
    -> need to relate the visual fouling to some data on heat exchanger efficiency.
    -> for numerical thresholds, we should look into:
            > temperature difference monitoring, calculating heat transfer coefficients.
            > infared imaging. This can identify areas causing blockage because heat is not evenly distributed.
            > measuring fouling resistance. The increase in thermal resistance due to fouling.
    -> we need some kind of temperature data in order to accurately determine correlation. Somehow we need to find efficiency.

- Additionally, I should edit the camera recording script and restart it so we can get more data (just in case).

NEW IDEA:
- Perhaps we can model fouling by creating a system that can predict using any heat transfer coefficients. 
- This way anyone could apply our research to their own individual setup.
- Also allows us to not need temperature data. We can just use coeeficients from another paper to validate our model.








# -------- ICML Notes -------- #

Experiment 1:
- Compare VD (Vignette-Dilation) and RET (Root-Expansion-Tiling) approaches using candlstick graph.
- Ensure full confusion matrix is tracked for testing binary models.
    -> Ensure this works for any plots and CSV files that get saved


- Problem: SMS datasets only consider a single hour on day 20, 40 and 60! Please fix!
- Generally need to clean up all ICML related code...








