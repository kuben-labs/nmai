Tripletex — Scoring
Field-by-Field Verification (Correctness)
After your agent responds, we query the Tripletex API to verify what was created or modified. Each task has specific checks worth different point values.
Example for a "Create employee" task (max 10 points):



Check
Points




Employee found
2


Correct first name
1


Correct last name
1


Correct email
1


Administrator role assigned
5



The raw score is normalized to 0–1: correctness = points_earned / max_points (e.g., 8/10 = 0.8).
Tier Multiplier
Each task has a difficulty tier that multiplies your correctness score:



Tier
Multiplier
Example tasks




Tier 1
×1
Create employee, create customer


Tier 2
×2
Create invoice, register payment


Tier 3
×3
Complex multi-step workflows



So a perfect score on a Tier 2 task = 1.0 × 2 = 2.0 base score.
Efficiency Bonus
If your agent achieves a perfect correctness score (1.0), you receive an efficiency bonus that can up to double your tier score.
Two factors determine the bonus:
Call efficiency — How many API calls did your agent make compared to the best known solution for this task? Fewer calls = higher bonus.
Error cleanliness — How many of your API calls resulted in 4xx errors (400, 404, 422, etc.)? Errors reduce the bonus. An agent that gets it right without trial-and-error is rewarded.



Scenario (Tier 2 task)
Score




Failed all checks
0.0


80% of checks passed
1.6


Perfect, but many errors and extra calls
~2.1


Perfect, efficient, a few errors
~2.6


Perfect, best-in-class efficiency, zero errors
4.0



The efficiency bonus only applies to perfect submissions. Non-perfect submissions score correctness × tier.
Efficiency benchmarks are recalculated periodically. As teams find more efficient solutions, the bar rises for everyone. Your best score per task is recalculated against current benchmarks every 12 hours.
Best Score Per Task
Your score per task is your all-time best. Bad runs never lower your score — only improvements count.

One good run is enough to lock in a score
You can always improve by submitting again
Focus on building a better agent, not grinding to recover from bad luck
Each of the 30 tasks tracks independently

Leaderboard
Total leaderboard score = sum of best scores across all task types.
The more task types your agent handles well, the higher your potential score.
Task Assignment
Each submission receives one task, weighted toward tasks you've attempted less. Over many submissions, you'll encounter all task types. Tasks are grouped into three tiers:

Tier 1 — foundational tasks (e.g., create employee, create customer, create invoice)
Tier 2 — multi-step workflows (e.g., invoice with payment, credit notes, project billing)
Tier 3 — complex scenarios (e.g., bank reconciliation from CSV, error correction in ledger, year-end closing)

Each task has 56 unique variants (7 languages × 8 data sets), so you'll rarely see the same prompt twice.
Tier Release Schedule
Tasks are released in tiers throughout the competition:

Tier 1 — available from competition start
Tier 2 — opens early Friday. Check this page for updates.
Tier 3 — opens early Saturday. Check this page for updates.

This gives you time to build a solid agent on simpler tasks before tackling the harder ones.
Rate Limits



Limit
Verified teams
Unverified teams




Concurrent submissions
3
1


Per task per day
5
2


