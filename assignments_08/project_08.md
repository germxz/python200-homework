# Step 5

I successfully set up my Supabase project and created both required tables — weather_raw and weather_enriched are visible in the Table Editor with the correct columns. Note: re-running the CREATE TABLE statement a second time throws an "already exists" error, which is expected since the table was already created successfully on the first run.

# Part B: Cloud Cost Analysis

Scenario A:
MONTHLY - $1.66
YEARLY - $19.92

A t3.micro EC2 instance (on-demand, US East N. Virginia) running about 160 hours per month costs $1.66 per month, or $19.92 per year. This surprised me with how cheap it is — a light, part-time workload like this costs about the same per year as a couple of lunches, which makes it very approachable for an indie developer.

Scenario B:
MONTHLY - $2,580.38
YEARLY - $30,964.56

A p3.2xlarge GPU instance running 24/7 for the full month (730 hours), plus an RDS db.m5.large instance and 1 TB of S3 Standard storage (all in US East N. Virginia), costs $2,580.38 per month, or $30,964.56 per year. This matched what I expected — running a GPU around the clock is expensive, and the GPU instance accounts for almost the entire bill.

Scenario A's yearly cost surprised me by how cheap it is for an indie developer, while Scenario B's cost matched what I expected from running a GPU continuously. I also found it interesting how many different EC2 instance families exist, each offering different amounts of compute, memory, and GPU power — there are far more configuration options than I expected, and I'd like to explore more of them.
Scenario A only costs money while it's actually being used, whereas Scenario B runs 24/7 regardless of traffic, which is what drives its bill so much higher. A GPU instance like this is only worth that cost when a workload genuinely needs sustained, heavy processing power constantly, such as training a machine learning model or running large-scale analytics continuously — otherwise, that expense would be paying too much.

Video Link: https://youtu.be/vVR07kzt6fY
