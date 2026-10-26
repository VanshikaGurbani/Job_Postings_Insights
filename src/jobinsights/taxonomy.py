"""Skill taxonomy: canonical name, family, and the regex aliases that detect it.

Matching is word-bounded and case-insensitive unless the pattern wraps itself in
``(?-i:...)``. Ambiguous one-letter or common-word skills (R, Go, Excel, SAS, SAP) are
case-sensitive and need context, which is exactly where the original substring
matcher went wrong.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Skill:
    name: str
    family: str
    patterns: tuple[str, ...]

    def regex(self) -> re.Pattern[str]:
        body = "|".join(self.patterns)
        return re.compile(rf"(?<![A-Za-z0-9_+#])(?:{body})(?![A-Za-z0-9_+#])", re.IGNORECASE)


def _s(name: str, family: str, *patterns: str) -> Skill:
    return Skill(name, family, patterns or (re.escape(name),))


_LANGS = r"(?:Python|SAS|SPSS|Stata|MATLAB|SQL|Julia|Scala)"

SKILLS: tuple[Skill, ...] = (
    # Programming
    _s("Python", "Programming", r"python"),
    _s(
        "R",
        "Programming",
        r"(?-i:R)\s*(?:programming|language|studio|shiny|markdown)\b",
        r"rstudio",
        rf"(?-i:R(?=\s?[,/&]\s?{_LANGS}\b)|R(?=\s(?:and|or)\s{_LANGS}\b))(?!&)",
        rf"(?-i:{_LANGS}(?:,\s?|\s?/\s?|\s+and\s+|\s+or\s+)R)(?![&/])",
        r"(?-i:(?<=\bin )R|(?<=\busing )R)(?=\s?[,.;)])",
    ),
    _s("SQL", "Programming", r"sql(?!\s*server)", r"t-?sql", r"pl/?sql"),
    _s("Java", "Programming", r"java"),
    _s("JavaScript", "Programming", r"javascript", r"node\.?js", r"react(?:\.js|js)?"),
    _s("TypeScript", "Programming", r"typescript"),
    _s("C++", "Programming", r"c\+\+"),
    _s("C#", "Programming", r"c#", r"\.net"),
    _s("Go", "Programming", r"golang", r"(?-i:Go)\s+(?:programming|language)"),
    _s("Scala", "Programming", r"scala"),
    _s("Ruby", "Programming", r"ruby(?:\s+on\s+rails)?"),
    _s("PHP", "Programming", r"php"),
    _s("MATLAB", "Programming", r"matlab"),
    _s("SAS", "Programming", r"(?-i:SAS)"),
    _s("VBA", "Programming", r"vba", r"visual\s+basic"),
    _s("Shell/Bash", "Programming", r"bash", r"shell\s+script(?:ing)?", r"powershell"),
    _s("HTML/CSS", "Programming", r"html5?", r"css3?"),
    # Data science and ML
    _s("Machine Learning", "Data & ML", r"machine[\s-]+learning", r"(?-i:ML)"),
    _s("Deep Learning", "Data & ML", r"deep[\s-]+learning", r"neural\s+networks?"),
    _s("NLP", "Data & ML", r"nlp", r"natural[\s-]+language\s+processing"),
    _s("Computer Vision", "Data & ML", r"computer\s+vision"),
    _s(
        "Generative AI/LLM",
        "Data & ML",
        r"generative\s+ai",
        r"genai",
        r"large\s+language\s+models?",
        r"(?-i:LLMs?)",
    ),
    _s("Statistics", "Data & ML", r"statistics", r"statistical\s+(?:analysis|modeling|methods?)"),
    _s("Data Analysis", "Data & ML", r"data\s+analy(?:sis|tics|zing)", r"analy[sz]e\s+data"),
    _s("Data Visualization", "Data & ML", r"data\s+visuali[sz]ation", r"dashboards?"),
    _s("Data Modeling", "Data & ML", r"data\s+model(?:l)?ing"),
    _s("ETL", "Data & ML", r"etl", r"data\s+pipelines?"),
    _s("A/B Testing", "Data & ML", r"a/b\s+test(?:ing|s)?", r"experimentation"),
    _s("Forecasting", "Data & ML", r"forecasting"),
    _s("TensorFlow", "Data & ML", r"tensorflow"),
    _s("PyTorch", "Data & ML", r"pytorch"),
    _s("scikit-learn", "Data & ML", r"scikit[\s-]?learn", r"sklearn"),
    _s("pandas", "Data & ML", r"pandas"),
    _s("Spark", "Data & ML", r"(?:apache\s+)?spark", r"pyspark"),
    _s("Hadoop", "Data & ML", r"hadoop"),
    _s("Kafka", "Data & ML", r"kafka"),
    _s("Airflow", "Data & ML", r"airflow"),
    _s("dbt", "Data & ML", r"dbt"),
    # Business intelligence
    _s("Tableau", "BI", r"tableau"),
    _s("Power BI", "BI", r"power\s?bi"),
    _s("Excel", "BI", r"(?-i:Excel)(?!\s+(?:in|at|as|under)\b)", r"(?-i:MS|Microsoft)\s+Excel"),
    _s("Looker", "BI", r"looker"),
    _s("Alteryx", "BI", r"alteryx"),
    _s("Qlik", "BI", r"qlik(?:view|sense)?"),
    _s("Google Analytics", "BI", r"google\s+analytics"),
    # Databases and warehouses
    _s("PostgreSQL", "Databases", r"postgres(?:ql)?"),
    _s("MySQL", "Databases", r"mysql"),
    _s("SQL Server", "Databases", r"sql\s+server", r"ms\s?sql"),
    _s("Oracle DB", "Databases", r"oracle\s+(?:database|db)"),
    _s("MongoDB", "Databases", r"mongo(?:db)?"),
    _s("NoSQL", "Databases", r"nosql", r"dynamodb", r"cassandra"),
    _s("Redis", "Databases", r"redis"),
    _s("Snowflake", "Databases", r"snowflake"),
    _s("Redshift", "Databases", r"redshift"),
    _s("BigQuery", "Databases", r"big\s?query"),
    _s("Databricks", "Databases", r"databricks"),
    # Cloud and engineering
    _s("AWS", "Cloud & DevOps", r"aws", r"amazon\s+web\s+services"),
    _s("Azure", "Cloud & DevOps", r"azure"),
    _s("GCP", "Cloud & DevOps", r"gcp", r"google\s+cloud(?:\s+platform)?"),
    _s("Docker", "Cloud & DevOps", r"docker"),
    _s("Kubernetes", "Cloud & DevOps", r"kubernetes", r"k8s"),
    _s("Terraform", "Cloud & DevOps", r"terraform"),
    _s("CI/CD", "Cloud & DevOps", r"ci\s?/\s?cd", r"jenkins", r"github\s+actions"),
    _s("Git", "Cloud & DevOps", r"git(?:hub|lab)?"),
    _s("Linux", "Cloud & DevOps", r"linux", r"unix"),
    _s("REST APIs", "Cloud & DevOps", r"rest(?:ful)?\s+apis?"),
    # Business applications
    _s(
        "Microsoft Office",
        "Business tools",
        r"microsoft\s+office",
        r"ms\s+office",
        r"office\s+365",
        r"microsoft\s+365",
        r"powerpoint",
    ),
    _s("Salesforce", "Business tools", r"salesforce"),
    _s("SAP", "Business tools", r"(?-i:SAP)"),
    _s("QuickBooks", "Business tools", r"quickbooks"),
    _s("Workday", "Business tools", r"workday"),
    _s("HubSpot", "Business tools", r"hubspot"),
    _s("Jira", "Business tools", r"jira", r"confluence"),
    _s("Adobe Creative Suite", "Business tools", r"adobe\s+creative", r"photoshop", r"indesign"),
    _s("ERP", "Business tools", r"erp", r"netsuite"),
    _s("CRM", "Business tools", r"crm"),
    # Business and domain skills
    _s("Project Management", "Business", r"project\s+management", r"pmp"),
    _s("Agile/Scrum", "Business", r"agile", r"scrum", r"kanban"),
    _s("Budgeting", "Business", r"budget(?:ing)?s?", r"budget\s+management"),
    _s(
        "Financial Analysis", "Business", r"financial\s+(?:analysis|modeling|reporting|statements?)"
    ),
    _s("Accounting", "Business", r"accounting", r"gaap", r"general\s+ledger", r"reconciliations?"),
    _s("Auditing", "Business", r"audit(?:ing)?", r"sox"),
    _s("Customer Service", "Business", r"customer\s+service", r"customer\s+support"),
    _s(
        "Sales",
        "Business",
        r"sales\s+(?:experience|skills|techniques|process|presentations?|targets?)",
        r"(?:b2b|inside|outside)\s+sales",
        r"cold\s+calling",
    ),
    _s("Negotiation", "Business", r"negotiat(?:ion|ing|e)"),
    _s("Marketing", "Business", r"marketing", r"seo", r"social\s+media"),
    _s("Supply Chain", "Business", r"supply\s+chain", r"logistics", r"inventory\s+management"),
    _s(
        "Lean/Six Sigma",
        "Business",
        r"six\s+sigma",
        r"lean\s+(?:manufacturing|methodolog(?:y|ies)|principles|six\s+sigma)",
        r"process\s+improvement",
    ),
    _s("Recruiting", "Business", r"recruit(?:ing|ment)", r"talent\s+acquisition"),
    _s("Compliance", "Business", r"compliance", r"regulatory"),
    _s("Data Entry", "Business", r"data\s+entry"),
    # Healthcare and licences
    _s("HIPAA", "Healthcare", r"hipaa"),
    _s("CPR/BLS", "Healthcare", r"cpr", r"bls", r"basic\s+life\s+support"),
    _s(
        "Electronic Health Records",
        "Healthcare",
        r"electronic\s+(?:health|medical)\s+records?",
        r"ehr",
        r"emr",
    ),
    _s("Patient Care", "Healthcare", r"patient\s+care"),
    _s("Driver's License", "Credentials", r"driver'?s?\s+licen[sc]e", r"cdl"),
    # Soft skills
    _s(
        "Communication",
        "Soft skills",
        r"communication\s+skills?",
        r"(?:written|verbal|oral)\s+communication",
    ),
    _s("Leadership", "Soft skills", r"leadership"),
    _s("Teamwork", "Soft skills", r"teamwork", r"team\s+player", r"collaborat(?:e|ion|ive)"),
    _s("Problem Solving", "Soft skills", r"problem[\s-]+solving"),
    _s("Time Management", "Soft skills", r"time\s+management", r"multi-?task(?:ing)?"),
    _s("Attention to Detail", "Soft skills", r"attention\s+to\s+detail", r"detail[\s-]+oriented"),
    _s("Analytical Skills", "Soft skills", r"analytical", r"critical\s+thinking"),
    _s("Organization", "Soft skills", r"organi[sz]ational\s+skills?", r"well[\s-]+organi[sz]ed"),
)

SKILL_NAMES: tuple[str, ...] = tuple(s.name for s in SKILLS)
FAMILY: dict[str, str] = {s.name: s.family for s in SKILLS}
_REGEXES: tuple[tuple[str, re.Pattern[str]], ...] = tuple((s.name, s.regex()) for s in SKILLS)

# Common free-text spellings (LLM output, skills_desc) that the regexes do not cover as a bare
# token, mapped to canonical names. Lower-cased keys.
EXTRA_ALIASES: dict[str, str] = {
    "ml": "Machine Learning",
    "llm": "Generative AI/LLM",
    "llms": "Generative AI/LLM",
    "powerbi": "Power BI",
    "ms excel": "Excel",
    "microsoft excel": "Excel",
    "postgres": "PostgreSQL",
    "js": "JavaScript",
    "node": "JavaScript",
    "golang": "Go",
    "ms office": "Microsoft Office",
    "word": "Microsoft Office",
    "powerpoint": "Microsoft Office",
    "communication": "Communication",
    "customer support": "Customer Service",
    "team work": "Teamwork",
    "collaboration": "Teamwork",
    "detail oriented": "Attention to Detail",
    "analytical": "Analytical Skills",
    "organizational skills": "Organization",
}


def normalize_skill(text: str) -> str | None:
    """Map a free-text skill mention to a canonical taxonomy name, or None if unknown."""
    cleaned = re.sub(r"\s+", " ", text.strip())
    if not cleaned:
        return None
    key = cleaned.lower()
    for name in SKILL_NAMES:
        if name.lower() == key:
            return name
    if key in EXTRA_ALIASES:
        return EXTRA_ALIASES[key]
    for name, rx in _REGEXES:
        if rx.fullmatch(cleaned):
            return name
    return None


def extract_skills(text: str) -> list[str]:
    """Canonical skills mentioned in `text`, in taxonomy order."""
    return [name for name, rx in _REGEXES if rx.search(text)]


# The 18 keywords of the original notebook, matched the way it matched them
# (case-insensitive substring), kept only so the evaluation can quantify the old method.
LEGACY_KEYWORDS = (
    "Python", "SQL", "Machine Learning", "Deep Learning", "AWS", "Azure", "Data Analysis",
    "Statistics", "NLP", "TensorFlow", "PyTorch", "Excel", "Leadership", "Communication",
    "Project Management", "R", "Tableau", "BigQuery",
)  # fmt: skip


def legacy_skills(text: str) -> list[str]:
    low = text.lower()
    return [k for k in LEGACY_KEYWORDS if k.lower() in low]
