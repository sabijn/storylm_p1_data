# Logboek human evaluation

## Get stories to Qualtrics
### First version
In the first version, all 200 stories were embedded in the JS of Qualtrics These were sampled per participant. With this approach there was no control over the sampling and over-sampling is needed to ensure that all stories get 3 ratings. 

### Second version
A pre-assignment is made over all 60 participants. These assignments are stored in a google sheet together with a counter and an App script wrapper. This wrapper is called by a webserver in Cloudflare (because google sheets does redirects and Qualtrics doesn't work with redirects). The assignments only contain indices. The stories are still stored in JS of Qualtrics. 

## Missing IDs
After the qualtrics results came in, a few IDs (9) were missing. Probably because people started simultaneously or returned their submission mid-experiment. These IDs were identified with `human_results_processing.ipynb` and pasted in the google sheets. Counter is reset and the link for the qualtrics is shared with people in the surrounding to get the last judgements.  

Remember that you reset the counter and therefore the ids of these missing ones. In the Qualtrics export they will have the wrong ID. You can map this back based on the `assignments_sheets_missing_ids.csv' as the Qualtrics rows contain the 10 sampled ids. 