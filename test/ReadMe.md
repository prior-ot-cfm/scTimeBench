# Testing Framework

## End-to-end testing
When testing end-to-end we need to ensure that the following works:
1. We create a new model output directory for each model type that we create, i.e. per dataset, and per dataset x filters as well.
2. We need to test that we are getting the expected output from the run.py as well per model, and that each model being integrated works as expected.
3. We need to test that the datasets, dataset filters, etc. are populated as expected.
4. We need to ensure that running the higher level metrics work, so we can group these metrics together, and that they end up with the same caching location if needed.
5.

## Trajectory Inference Module
This requires its own separate testing to make sure that the implementations are working as expected.
1. Create your own 2D data that is easy to reason about.
2. Run it through the trajectory inference module and make sure that it works.

## Metrics
We need to test each metric that we create and ensure that they work as expected.
