**DeFi Credit Scoring Analysis Report**

Date: 17-07-2025
Data Source: Aave V2 Protocol
Wallets Analyzed: 3,497(unique)
Score Range: 50 – 950
Mean Score: 650 | Median Score: 695

**Executive Summary**

This report analyzes DeFi credit scores for 3,497 wallet addresses, leveraging transactional data from Aave V2. The credit scores are constructed using a hybrid system of anomaly detection, behavioral clustering, and rule-based financial scoring. Final scores are normalized to a 0–1000 range and grouped into five risk tiers.

**Scoring Methodology**

-> Anomaly Detection (0–300 pts): Identifies suspicious transaction patterns via Isolation Forest

-> Behavioral Clustering (0–250 pts): Uses K-Means to group users by similar activity

-> Rule-Based Scoring (0–450 pts): Scores based on financial behaviors and risk indicators

-> Final Score Range: 0–1000

   # Excellent: 800–1000

   # Good: 700–799

   # Fair: 600–699

   # Poor: 500–599

   # Very Poor: 0–499

**Score Distribution Overview**

      Tier	                   Score Range                Count	                  % Share	                 Description
   Excellent	                800–1000	              224                     6.4%	                   Premium users
    Good	                    700–799                   618	                  17.7%                 Low-risk,reliable users
    Fair	                    600–699	                 1,865	                  53.3%          Standard users with acceptable risk
    Poor	                    500–599	                  313	                  9.0%             High-risk, needs monitoring
  Very Poor                   	0–499	                  477	                  13.6%	          Critical risk, restricted access

*Key Stats:*

Median > Mean: Indicates right-skewed score distribution

High Risk Users (0–599): 22.6%

Premium Users (700+): 24.1%

Majority in Fair Tier: 53.3%


**Key Insights**

-> Healthy Mid-Tier Base: 53.3% in Fair range offers upgrade potential

-> Valuable Minority: 24.1% of users score 700+, indicating low risk and high value

-> High-Risk Segments: 13.6% in Very Poor tier require proactive attention

-> Behavioral Validity: Clear, predictable user patterns align with score ranges

-> Business Potential: Tiered structure enables pricing, access, and targeting strategies
