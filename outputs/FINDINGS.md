All figures come from 123,782 postings; pay figures use the 3,428 data, AI and software postings that state a salary (1,411 companies, median $133,950). Pay comparisons are adjusted for role, experience level and work arrangement and are associations, not causal effects.

**1. Which skills go with higher pay.** After adjusting, the skills most associated with higher pay are TypeScript +16% (+8 to +21), Generative AI/LLM +15% (+2 to +29), Machine Learning +13% (+5 to +20), Go +12% (+3 to +22) (95% interval in brackets). Python itself shows +1% (-2 to +7) because it is listed so widely that it does not separate higher- from lower-paid postings. Excel is associated with -9% (-14 to -5). Lower for TensorFlow (-12%) and C# (-10%).

![Skill pay association](docs/figures/skill_premium.png)

**2. Remote pays no clearly different amount.** Remote postings are +6.3% vs onsite (-2.7 to +21.2) after the same adjustment; the interval includes zero, so the data cannot separate a remote premium from no difference.

**3. More data postings pass a sponsorship screen than the average posting.** 42% of Data Analyst postings are not restricted and either say sponsorship is possible or come from a company with H-1B approvals in FY2021-23, against 26% of other postings. Only 6% of Data Analyst postings rule sponsorship out and only 3% say it is offered, so for most the signal is the company's history, not the posting. That history says a company files petitions, not that this role would.

![Sponsor share by role](docs/figures/sponsor_share.png)

![Employers](docs/figures/top_employers.png)

_Approval counts cover only the legal-entity names the company matched to, so parent companies whose petitions are filed under other names are undercounted (see Limitations)._

**4. Data roles skew senior.** 23% of Data Analyst postings are entry-level, associate or internship, against 40% of other postings, so entry-level data jobs are a smaller pool to search.

![Data analyst skills](docs/figures/data_analyst_skills.png)

**5. What to learn first.** Across 200 entry-level Data Analyst postings that list at least two technical skills, learning Data Analysis, SQL, Excel, Data Visualization, Python in that order gets you to at least half of the listed skills in 65% of them (from 0% with none). Order is chosen greedily and describes skill overlap with postings, not hiring odds. The dashboard has the same planner for your own skill set.

![Learning path](docs/figures/learning_path.png)