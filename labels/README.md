# Labels

Three JSON files used by the training and evaluation scripts.

## `class_labels.json`

Class-keyed mapping `{class_name: [list of training-time response variants]}`.
Used as a fallback by the dataset class when a file-keyed label is missing.

The 11 classes:

| Class | Exercise | Fault described |
|---|---|---|
| `squat_no_errors`       | squat | clean rep, no fault |
| `squat_butt_wink`       | squat | pelvis tucks under at bottom, lumbar rounds |
| `squat_depth_high`      | squat | too shallow / femurs above parallel |
| `squat_hands_wide`      | squat | grip too wide on the bar |
| `squat_head_position`   | squat | head/neck out of neutral |
| `squat_high_heel`       | squat | heels lift off the floor |
| `rdl_no_error`          | Romanian deadlift | clean rep |
| `rdl_hands_forward`     | RDL | bar drifts away from the legs |
| `rdl_too_much_depth`    | RDL | descent passes the hip's hinge range |
| `rdl_too_much_knee_bend`| RDL | knees bend too much, exercise becomes a squat |
| `rdl_head_position`     | RDL | head not following torso |

## `labels_5var_reusable.json`

File-keyed mapping `{video.mp4: [5 response variants]}`. Used as the primary
training target — for each video, one of the 5 variants is randomly sampled
per epoch.

## `class_labels_pooled_150.json`

Class-keyed mapping `{class_name: [150 response variants]}`. Used by the
BERTScore classifier at evaluation time as the reference pool — each
response is scored against a random sub-sample of these references and the
top-scoring class is the predicted class.
