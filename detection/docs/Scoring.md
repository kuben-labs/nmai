NorgesGruppen Data: Scoring
Hybrid Scoring
Your final score combines detection and classification:
Score = 0.7 × detection_mAP + 0.3 × classification_mAP

Both components use mAP@0.5 (Mean Average Precision at IoU threshold 0.5).
Detection mAP (70% of score)
Measures whether you found the products, ignoring category:

Each prediction is matched to the closest ground truth box
A prediction is a true positive if IoU ≥ 0.5 (category is ignored)
This rewards accurate bounding box localization

Classification mAP (30% of score)
Measures whether you identified the correct product:

A prediction is a true positive if IoU ≥ 0.5 AND the category_id matches the ground truth
356 product categories (IDs 0-355) from the training data annotations.json

Detection-Only Submissions
If you set category_id: 0 for all predictions, you can score up to 0.70 (70%) from the detection component alone. Adding correct product identification unlocks the remaining 30%.

Score range: 0.0 (worst) to 1.0 (perfect)

Submission Limits



Limit
Value




Submissions in-flight
2 per team


Submissions per day
3 per team


Infrastructure failure freebies
2 per day (don't count against your 3)



Limits reset at midnight UTC. If you hit an infrastructure error (our fault), it doesn't count against your daily limit — up to 2 per day. After that, infrastructure failures consume a regular submission slot.
Leaderboard
The public leaderboard shows scores from the public test set. The final ranking uses the private test set which is never revealed to participants.
Select for Final Evaluation
By default, your best-scoring submission is used for the final private evaluation. You can override this by clicking Select for final on any completed submission in your submission history. This lets you choose a submission you trust, even if it's not your highest public score. You can change your selection at any time before the competition ends.