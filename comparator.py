"""
Local AI resume vs JD comparator — ATS-grade matching.
Uses normalizer.py for all word-boundary-safe search and canonicalization.

Matching pipeline (in order):
  1. Exact / alias match   → score 1.0   (literal string in resume)
  2. Inference match       → score 0.90  (EKS → Kubernetes, "root-cause" → Dive Deep, etc.)
  3. Multi-query semantic  → cosine-sim of (raw + rich + verb-expanded queries)
                             boosted by token overlap
  4. TF-IDF fallback       → TF-IDF cosine + token overlap (no model needed)

100 % offline — no external API calls.
"""
import re
from normalizer import wcontains, check_trigger, all_variants, canonicalize, strip_version

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEMANTIC_THRESHOLD = 0.38   # min to count as partial
CLOSE_THRESHOLD    = 0.50   # above this = "close match"
MODEL_NAME         = "all-MiniLM-L6-v2"
INFER_SCORE        = 0.90

CATEGORY_WEIGHTS = {
    "tech_skills":       35,
    "responsibilities":  25,
    "amazon_lps":        20,
    "soft_skills":       12,
    "behavioral_traits":  8,
}

CATEGORY_LABELS = {
    "tech_skills":       "Technical Skills",
    "responsibilities":  "Key Responsibilities",
    "amazon_lps":        "Amazon Leadership Principles",
    "soft_skills":       "Soft Skills",
    "behavioral_traits": "Behavioral Traits",
}

# ---------------------------------------------------------------------------
# Alias map — exact-string synonyms
# ---------------------------------------------------------------------------

_ALIASES: dict[str, list[str]] = {
    "postgresql":                  ["postgres", "psql", "pg", "aurora postgres", "rds postgres"],
    "mongodb":                     ["mongo"],
    "kubernetes":                  ["k8s", "kube", "kubectl"],
    "elasticsearch":               ["elastic", "opensearch"],
    "javascript":                  ["js", "es6", "ecmascript", "es2015", "es2020"],
    "typescript":                  ["ts"],
    "node.js":                     ["nodejs", "node js", "node"],
    "react.js":                    ["react", "reactjs", "react js"],
    "next.js":                     ["nextjs", "next js"],
    "vue.js":                      ["vue", "vuejs"],
    "amazon web services":         ["aws"],
    "google cloud":                ["gcp", "google cloud platform"],
    "microsoft azure":             ["azure"],
    "ci/cd":                       ["continuous integration", "cicd", "ci cd", "continuous delivery"],
    "machine learning":            ["ml"],
    "natural language processing": ["nlp"],
    "large language model":        ["llm"],
    "retrieval augmented generation": ["rag"],
    "sql server":                  ["mssql", "ms sql server"],
    "apache kafka":                ["kafka"],
    "apache spark":                ["spark", "pyspark"],
    "github actions":              ["gha"],
    "docker":                      ["containerization", "container", "docker compose"],
    "infrastructure as code":      ["iac", "terraform", "pulumi", "cdk"],
    "terraform":                   ["iac", "infrastructure as code"],
    "restful":                     ["rest", "rest api", "restful api", "http api"],
    "object-oriented":             ["oop", "object oriented", "object-oriented programming"],
    "test-driven development":     ["tdd"],
    "redis":                       ["elasticache", "redis cache", "in-memory cache"],
    "mysql":                       ["mariadb", "aurora mysql", "rds mysql"],
    "graphql":                     ["apollo", "graphql api", "graphql schema"],
    "apache airflow":              ["airflow"],
    "databricks":                  ["delta lake"],
    "aws lambda":                  ["serverless function", "lambda function"],
    "dynamodb":                    ["dynamo db", "dynamo"],
    "microservices":               ["micro-services", "micro services", "service-oriented"],
    "agile":                       ["scrum", "kanban", "sprint", "standup"],
    "linux":                       ["unix", "ubuntu", "debian", "centos", "bash scripting"],
    "git":                         ["github", "gitlab", "bitbucket", "version control"],
}

_ALIAS_LOOKUP: dict[str, set[str]] = {}
for _canon, _alts in _ALIASES.items():
    _group = {_canon} | set(_alts)
    for _term in _group:
        if _term not in _ALIAS_LOOKUP:
            _ALIAS_LOOKUP[_term] = _group
        else:
            _ALIAS_LOOKUP[_term] |= _group

# ---------------------------------------------------------------------------
# Inference map — if resume contains ANY of these terms, parent skill implied
# ---------------------------------------------------------------------------

_INFER_FROM: dict[str, list[str]] = {
    # ── Technologies ────────────────────────────────────────────────────────
    "kubernetes": [
        "eks", "gke", "aks", "openshift", "helm chart", "kubectl",
        "container orchestration", "k8s cluster", "kubernetes cluster",
        "pod deployment", "ingress controller",
    ],
    "docker": [
        "dockerfile", "docker-compose", "docker compose",
        "containerized", "container image", "docker image",
        "docker swarm", "docker hub", "docker registry",
    ],
    "python": [
        "fastapi", "django", "flask", "pandas", "numpy",
        "pytorch", "tensorflow", "keras", "pyspark", "jupyter",
        "pytest", "sqlalchemy", "celery", "asyncio", "aiohttp",
        "scikit-learn", "sklearn", "boto3", "langchain",
    ],
    "javascript": [
        "react", "reactjs", "angular", "vue", "node.js", "nodejs",
        "next.js", "nextjs", "express.js", "expressjs",
        "webpack", "babel", "typescript",
    ],
    "typescript": ["tsx", "angular", "nestjs", "next.js"],
    "aws": [
        "ec2", "s3 bucket", "aws lambda", "lambda function",
        "rds", "dynamodb", "eks cluster", "ecs", "cloudformation",
        "cloudwatch", "sqs", "sns", "iam role", "vpc", "route 53",
        "api gateway", "aws glue", "redshift", "aws cdk", "cdk",
        "aws fargate", "fargate", "aws step functions", "aurora",
    ],
    "gcp": [
        "bigquery", "gke", "cloud run", "cloud functions",
        "pub/sub", "dataflow", "vertex ai", "cloud storage",
    ],
    "azure": [
        "azure devops", "azure functions", "aks cluster", "cosmos db",
        "azure blob", "azure sql", "azure ad", "azure pipelines",
    ],
    "postgresql": ["rds postgres", "aurora postgres", "neon", "supabase"],
    "mysql":      ["rds mysql", "aurora mysql", "mariadb"],
    "redis":      ["elasticache", "valkey", "upstash"],
    "machine learning": [
        "model training", "trained a model", "neural network",
        "deep learning", "gradient descent", "feature engineering",
        "ml pipeline", "pytorch", "tensorflow", "scikit-learn",
        "xgboost", "random forest", "fine-tuned", "fine tuning",
        "rag pipeline", "inference endpoint",
    ],
    "ci/cd": [
        "jenkins", "github actions", "gitlab ci", "circleci",
        "travis ci", "automated deployment", "build pipeline",
        "deployment pipeline", "release automation", "argocd",
        "spinnaker", "tekton", "continuous deployment",
    ],
    "terraform": [
        "infrastructure as code", "iac", "cloud provisioning", "pulumi",
        "terraform module", "terraform plan",
    ],
    "microservices": [
        "service mesh", "api gateway", "event-driven",
        "distributed system", "soa", "grpc", "protobuf",
        "monolith to microservices", "microservice architecture",
    ],
    "kafka": [
        "event streaming", "message queue", "confluent", "apache kafka",
        "event-driven architecture", "stream processing", "kinesis",
        "sqs", "sns", "rabbitmq", "pub/sub",
    ],
    "spark": [
        "pyspark", "spark job", "databricks", "spark cluster",
        "spark streaming", "spark sql", "emr",
    ],
    "agile": [
        "scrum", "sprint", "kanban", "jira", "standup",
        "retrospective", "backlog", "story points",
    ],
    "sql": [
        "database query", "stored procedure", "sql query", "joins",
        "indexing", "query optimization", "orm", "database schema",
    ],

    # ── Amazon Leadership Principles ────────────────────────────────────────
    "ownership": [
        "end-to-end", "owned the", "owned and", "sole owner",
        "single-threaded", "took responsibility", "led the initiative",
        "drove the project", "accountable for", "spearheaded",
        "took ownership", "full ownership",
    ],
    "deliver results": [
        "shipped", "launched", "deployed to production",
        "ahead of schedule", "weeks ahead", "days early",
        "on schedule", "on time", "met deadline", "delivered on",
        "reduced latency", "improved performance", "cut mttr",
        "increased throughput", "reduced by", "improved by",
        "% reduction", "% improvement", "% faster", "% increase",
        "% decrease", "cost savings", "saving", "saved",
    ],
    "dive deep": [
        "root cause analysis", "root-cause", "root cause investigation",
        "root cause", "deep investigation", "metrics dashboard",
        "data-driven", "performance profiling", "production debugging",
        "p0 incident", "p1 incident", "incident investigation",
        "cut mttr", "rca", "post-mortem", "postmortem",
        "investigated and", "traced the issue",
    ],
    "invent & simplify": [
        "automated", "simplified", "reduced toil", "innovative solution",
        "removed manual", "eliminated", "streamlined", "proactively built",
        "self-healing", "saving", "h/week", "hours per week", "toil",
        "novel approach", "invented", "created from scratch",
        "hours saved", "days saved", "proactively",
    ],
    "customer obsession": [
        "user experience", "customer satisfaction", "reduced friction",
        "improved retention", "nps", "user feedback", "customer impact",
        "customer-facing", "checkout experience", "end user",
        "user-facing", "customer pain point", "reduced churn",
    ],
    "bias for action": [
        "moved fast", "quick iteration", "shipped in days",
        "prototype", "rapid", "launched within", "fast-tracked",
        "without waiting", "preemptively", "proactively",
        "bias for speed", "moved quickly",
    ],
    "think big": [
        "long-term vision", "platform thinking", "strategic",
        "future-proof", "architectural vision", "company-wide",
        "org-wide", "multi-year", "north star", "roadmap",
        "zero to one", "platform play",
    ],
    "earn trust": [
        "built trust", "transparent", "cross-team alignment",
        "stakeholder communication", "post-mortem", "postmortem",
        "collaborated openly", "exec review", "monthly review",
    ],
    "frugality": [
        "reduced cost", "optimized spending", "cost savings",
        "efficient solution", "minimal resources", "infra cost",
        "cost reduction", "reduced infra", "budget",
    ],
    "hire & develop the best": [
        "mentored", "coached junior", "conducted interviews",
        "grew the team", "onboarded", "bar raiser",
        "promoted", "developed talent", "career growth",
    ],
    "insist on highest standards": [
        "code review", "test coverage", "zero downtime",
        "quality bar", "linting", "ci tests", "regression tests",
        "sla", "uptime", "reliability", "slo",
    ],
    "are right, a lot": [
        "data-driven decisions", "sound judgment", "evaluated tradeoffs",
        "recommended", "analysis-driven", "made the correct call",
    ],
    "have backbone; disagree & commit": [
        "challenged status quo", "proposed alternative", "drove adoption",
        "disagreed and committed", "pushed back", "influenced direction",
    ],
    "learn & be curious": [
        "quickly learned", "self-taught", "picked up", "explored",
        "research spike", "stayed current", "new technology",
    ],

    # ── Soft skills ─────────────────────────────────────────────────────────
    "leadership": [
        "led a team", "led the team", "managed engineers", "mentored",
        "tech lead", "engineering manager", "staff engineer",
        "team lead", "led the project", "led development",
        "led a squad", "led a group",
    ],
    "communication": [
        "stakeholder", "cross-functional", "presented to",
        "collaborated with", "written documentation", "technical writing",
        "weekly sync", "exec review", "monthly review",
        "communicated", "status update",
    ],
    "teamwork": [
        "collaborated with", "collaboration", "partnered with",
        "worked alongside", "cross-functional", "joint initiative",
        "team effort", "worked with", "coordinated with",
    ],
    "problem solving": [
        "root cause", "debugged", "diagnosed", "resolved an issue",
        "troubleshot", "investigated", "optimized performance",
        "fixed a bug", "resolved the incident", "cut mttr",
    ],

    # ── Behavioral traits ────────────────────────────────────────────────────
    "accountability": [
        "spearheaded", "drove", "owned the", "took full responsibility",
        "accountable", "single-threaded owner", "responsible for",
        "end-to-end", "took ownership", "delivered on",
    ],
    "initiative": [
        "proactively", "proactive", "self-starter", "proposed",
        "without being asked", "identified and resolved",
        "identified opportunity", "on own initiative",
        "self-initiated", "independently",
    ],
    "collaboration": [
        "collaborated across teams", "worked with product",
        "partnered with stakeholders", "joint initiative",
        "cross-team", "cross-functional collaboration",
        "worked alongside", "collaborated with",
    ],
    "mentoring": [
        "mentored junior engineers", "coached", "developed talent",
        "knowledge sharing", "pair programming", "tech talks",
        "promoted", "grew the team",
    ],
    "innovation": [
        "built novel solution", "invented", "innovative approach",
        "prototyped", "creative solution", "zero to one",
        "novel algorithm", "pioneered",
    ],
    "analytical": [
        "data-driven", "analysis", "metrics dashboard",
        "investigated patterns", "quantitative", "experiment design",
        "a/b test", "statistical",
    ],
    "critical thinking": [
        "evaluated tradeoffs", "analyzed options", "weighed pros and cons",
        "architectural decision", "trade-off analysis",
    ],
    "time management": [
        "delivered on time", "managed priorities", "met deadlines",
        "ahead of schedule", "multiple projects simultaneously",
        "completed ahead", "on schedule",
    ],
    "attention to detail": [
        "thorough review", "careful testing", "zero bugs",
        "high accuracy", "precise implementation",
    ],
    "adaptability": [
        "adapted to changes", "quickly learned new stack",
        "context switching", "pivoted technology",
    ],
}

# ---------------------------------------------------------------------------
# Action verb synonyms — used to expand JD responsibility queries
# ---------------------------------------------------------------------------

_RESP_VERB_SYNS: dict[str, list[str]] = {
    "design":     ["architect", "build", "develop", "create", "define", "engineer"],
    "architect":  ["design", "build", "define", "create", "develop"],
    "build":      ["develop", "create", "implement", "engineer", "write", "ship"],
    "develop":    ["build", "create", "implement", "engineer", "write"],
    "implement":  ["build", "develop", "create", "deploy", "ship", "write"],
    "lead":       ["own", "manage", "drive", "head", "oversee", "run", "spearhead"],
    "manage":     ["lead", "own", "oversee", "drive", "run", "head"],
    "own":        ["lead", "manage", "drive", "oversee", "spearhead"],
    "optimize":   ["improve", "enhance", "tune", "scale", "boost", "accelerate"],
    "improve":    ["optimize", "enhance", "upgrade", "increase", "boost"],
    "define":     ["design", "create", "establish", "set up", "determine", "establish"],
    "deliver":    ["ship", "launch", "release", "complete", "finish"],
    "deploy":     ["ship", "launch", "release", "roll out", "push", "deliver"],
    "scale":      ["grow", "expand", "optimize", "increase capacity"],
    "maintain":   ["support", "operate", "own", "manage", "run"],
    "mentor":     ["coach", "develop", "grow", "guide", "support"],
    "analyze":    ["investigate", "examine", "evaluate", "assess", "review"],
    "monitor":    ["observe", "track", "measure", "alert on", "watch"],
    "collaborate": ["partner with", "work with", "coordinate with"],
    "support":    ["help", "assist", "enable", "maintain"],
    "drive":      ["lead", "own", "manage", "spearhead", "push"],
}

# ---------------------------------------------------------------------------
# Semantic query expansions — rich descriptions for abstract items
# ---------------------------------------------------------------------------

_SEMANTIC_QUERIES: dict[str, str] = {
    # Amazon Leadership Principles
    "Ownership":
        "owned end-to-end delivery took full ownership single-threaded owner "
        "accountable for outcomes drove project from start to finish spearheaded "
        "responsible for results led the initiative",
    "Bias for Action":
        "shipped quickly moved fast launched MVP rapid delivery fast execution "
        "iterative development bias for speed prototype delivered in days weeks "
        "without waiting preemptively",
    "Deliver Results":
        "delivered results on time reduced latency improved performance shipped "
        "on schedule quantifiable outcomes measurable impact met deadline launched "
        "ahead of schedule % improvement % reduction cut costs",
    "Think Big":
        "long-term vision strategic thinking designed future-proof architecture "
        "scaled system platform thinking org-wide impact company-wide initiative "
        "north star roadmap multi-year plan",
    "Customer Obsession":
        "improved user experience customer satisfaction reduced friction improved "
        "retention nps user feedback customer impact customer-facing feature "
        "checkout experience end user",
    "Dive Deep":
        "root cause analysis root-cause investigation deep investigation "
        "metrics dashboard data-driven performance profiling production debugging "
        "p0 incident rca post-mortem cut mttr incident response",
    "Invent & Simplify":
        "automated process simplified system reduced toil innovative solution "
        "removed manual steps eliminated complexity streamlined workflow "
        "proactively built self-healing saving hours per week",
    "Have Backbone; Disagree & Commit":
        "challenged status quo proposed alternative approach drove adoption "
        "disagreed and committed pushed back influenced direction",
    "Are Right, A Lot":
        "data-driven decisions sound judgment recommended approach based on "
        "analysis made correct technical call evaluated tradeoffs",
    "Learn & Be Curious":
        "quickly learned new technology self-taught picked up new stack explored "
        "research prototype stayed current with industry trends",
    "Hire & Develop the Best":
        "mentored engineers grew team coached junior engineer conducted interviews "
        "developed talent onboarded new members bar raiser",
    "Insist on Highest Standards":
        "high code quality test coverage zero downtime code review bar raiser "
        "production reliability engineering excellence sla slo",
    "Earn Trust":
        "built trust transparent communication cross-team alignment stakeholder "
        "management post-mortem collaborated openly exec review",
    "Frugality":
        "reduced infrastructure costs optimized spending cost savings efficient "
        "solution minimal resources infra cost reduction budget",
    # Soft skills
    "communication":
        "communicated with stakeholders presented findings written documentation "
        "cross-functional collaboration verbal written exec review status update",
    "teamwork":
        "collaborated with team worked cross-functionally partnered with "
        "worked alongside team player coordinated joint initiative",
    "leadership":
        "led team managed engineers drove alignment technical lead mentored "
        "staff engineer principal engineer squad lead",
    "problem solving":
        "resolved issues debugged root cause analysis optimized performance "
        "investigated diagnosed fixed production incident cut mttr",
    "critical thinking":
        "analyzed trade-offs evaluated options data-driven recommendation "
        "weighed pros and cons architectural decision",
    "time management":
        "delivered on time managed priorities met deadlines ahead of schedule "
        "multiple projects completed simultaneously",
    "adaptability":
        "adapted to changes quickly learned new stack context switching "
        "pivoted technology evolved with requirements",
    "attention to detail":
        "thorough review careful testing zero bugs high accuracy precise",
    # Behavioral traits
    "accountability":
        "took responsibility owned the outcome accountable for results "
        "drove to completion accepted ownership spearheaded delivered on",
    "initiative":
        "proactively identified proposed solution self-starter "
        "drove improvement independently without being asked",
    "collaboration":
        "collaborated across teams worked with product design engineering "
        "partnered with stakeholders joint initiative cross-team",
    "mentoring":
        "mentored junior engineers coached developed talent knowledge sharing "
        "pair programming promoted grew the team",
    "innovation":
        "built novel solution invented new approach explored emerging tech "
        "prototyped creative solution pioneered",
    "analytical":
        "data-driven analysis metrics dashboards investigated patterns "
        "quantitative reasoning experiment design a/b test",
}

# ---------------------------------------------------------------------------
# Word-level inference — key verbs/nouns in a JD responsibility → resume evidence
# Used when the full item isn't in _INFER_FROM (catches dynamic responsibility phrases)
# ---------------------------------------------------------------------------

_WORD_INFER: dict[str, list[str]] = {
    # Responsibility action verbs / core themes
    "optimize":      ["% improvement", "% reduction", "% faster", "% decrease",
                      "query optimization", "tuned", "optimized", "latency reduction",
                      "performance improvement", "improved by"],
    "performance":   ["latency", "throughput", "rps", "qps", "response time",
                      "% faster", "benchmarked", "profiled", "p99", "p95"],
    "reliability":   ["sla", "slo", "mttr", "uptime", "zero-downtime",
                      "incident response", "high availability", "99.", "availability"],
    "scalable":      ["scaled", "scaling", "high throughput", "load testing",
                      "rps", "qps", "millions of requests", "high volume"],
    "scalability":   ["scaled", "scaling", "horizontal scaling", "load balanced",
                      "high throughput", "distributed"],
    "architecture":  ["architected", "system design", "designed the system",
                      "architectural", "architecture decision", "designed and built"],
    "architect":     ["architected", "system design", "designed the",
                      "architecture decision", "technical design"],
    "roadmap":       ["tech roadmap", "technical vision", "roadmap", "quarterly planning",
                      "product roadmap", "defined the roadmap"],
    "technical":     ["architected", "system design", "technical design",
                      "engineering", "backend", "infrastructure", "stack"],
    "distributed":   ["microservices", "distributed system", "service mesh",
                      "event-driven", "distributed architecture"],
    "security":      ["auth", "authentication", "authorization", "oauth",
                      "encryption", "tls", "ssl", "rbac", "iam"],
    "observability": ["monitoring", "alerting", "metrics", "tracing",
                      "logging", "datadog", "cloudwatch", "prometheus", "grafana"],
    "monitoring":    ["alerting", "metrics", "dashboards", "prometheus",
                      "grafana", "datadog", "cloudwatch", "pagerduty"],
    "infrastructure": ["terraform", "cloud", "kubernetes", "docker", "iac",
                       "provisioning", "vpc", "networking"],
    "platform":      ["platform engineering", "internal platform", "developer platform",
                      "platform team", "self-service", "shared infrastructure"],
    "api":           ["rest api", "graphql", "openapi", "swagger", "endpoint",
                      "api design", "api layer", "http api"],
    "data":          ["database", "pipeline", "analytics", "etl", "warehouse",
                      "data model", "schema", "postgres", "mysql"],
    "pipeline":      ["data pipeline", "etl", "stream processing", "batch job",
                      "airflow", "spark", "kafka", "kinesis"],
    "mentoring":     ["mentored", "coached", "grew the team", "developed talent",
                      "pair programming", "promoted"],
    "mentor":        ["mentored", "coached", "junior engineers", "promoted",
                      "knowledge sharing"],
    "testing":       ["unit tests", "integration tests", "test coverage",
                      "pytest", "jest", "tdd", "regression tests"],
    "deploy":        ["deployed to production", "deployment", "rollout",
                      "release", "ci/cd", "argocd", "github actions"],
    "migration":     ["migrated", "migration", "refactored", "rewrote",
                      "moved from", "upgraded"],
    "cross-functional": ["cross-functional", "product", "design", "pm",
                         "stakeholders", "collaborated with"],
    "collaboration": ["collaborated", "partnered", "cross-team", "joint",
                      "worked with"],
    "define":        ["defined", "established", "created", "set up",
                      "designed", "architected"],
}

# Stop words for token overlap
_STOPS = frozenset({
    "and", "or", "the", "a", "an", "of", "in", "for", "with", "to",
    "is", "are", "on", "at", "by", "as", "be", "was", "were", "that",
    "this", "it", "its", "from", "have", "has", "had", "will", "can",
    "their", "our", "your", "my", "we", "they", "you", "i", "he",
    "she", "not", "but", "if", "do", "did", "so", "up", "use", "used",
    "using", "via", "able", "across", "into", "over", "more", "new",
    "well", "both", "also", "such", "than", "then", "when", "where",
    "how", "which", "who", "what", "all", "any", "each", "other",
})

# ---------------------------------------------------------------------------
# Lazy-loaded models
# ---------------------------------------------------------------------------

_st_model   = None
_tfidf_vect = None


def _get_st_model():
    global _st_model
    if _st_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _st_model = SentenceTransformer(MODEL_NAME)
        except Exception:
            _st_model = False
    return _st_model


def _get_tfidf():
    global _tfidf_vect
    if _tfidf_vect is None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        _tfidf_vect = TfidfVectorizer(
            ngram_range=(1, 3),
            stop_words="english",
            min_df=1,
            sublinear_tf=True,
        )
    return _tfidf_vect

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _split_chunks(text: str) -> list[str]:
    """
    Split resume into chunks for semantic search.
    Keeps full bullet points + adds 2-line and 3-line sliding windows.
    Also prepends section headers to bullets for context.
    """
    raw_lines   = text.splitlines()
    lines       = []
    last_header = ""

    for raw in raw_lines:
        stripped = re.sub(r"^[-•*·▪◦➤→✦✓]\s*", "", raw).strip()
        stripped = re.sub(r"^\d+[.)]\s*", "", stripped).strip()
        if not stripped:
            continue
        is_header = (
            len(stripped) < 60
            and not stripped.startswith(("•", "-", "*"))
            and stripped.endswith((":", "—", "–"))
            or re.match(r"^[A-Z][A-Za-z\s&/]+$", stripped)
        )
        if is_header:
            last_header = stripped
        elif len(stripped) > 10:
            lines.append((last_header, stripped))

    if not lines:
        return [text[:1000]]

    chunks = []
    for header, line in lines:
        chunks.append(line)
        if header:
            chunks.append(f"{header}: {line}")

    # Sliding 2-line windows
    for i in range(len(lines) - 1):
        combined = lines[i][1] + " " + lines[i + 1][1]
        if len(combined) < 450:
            chunks.append(combined)

    # Sliding 3-line windows
    for i in range(len(lines) - 2):
        combined = " ".join(l for _, l in lines[i:i + 3])
        if len(combined) < 600:
            chunks.append(combined)

    return chunks


def _is_exact(item: str, resume_lower: str) -> bool:
    """
    Word-boundary safe exact check.
    Also checks all known aliases/variants of the item.
    Handles version normalization: "Python 3.9" matches resume saying "Python".
    """
    item_lower = item.lower()

    # Direct word-boundary check
    if wcontains(item_lower, resume_lower):
        return True

    # Check all alias variants from normalizer (covers postgres↔postgresql, etc.)
    for variant in all_variants(item):
        if wcontains(variant, resume_lower):
            return True

    # Legacy alias lookup (kept for backward compat with hand-crafted _ALIASES)
    for equiv in _ALIAS_LOOKUP.get(item_lower, set()):
        if wcontains(equiv, resume_lower):
            return True

    # Version-stripped match: "Python 3.9" → also check plain "Python"
    stripped = strip_version(item).strip()
    if stripped and stripped.lower() != item_lower:
        if wcontains(stripped.lower(), resume_lower):
            return True
        for variant in all_variants(stripped):
            if wcontains(variant, resume_lower):
                return True

    return False


def _is_implied(item: str, resume_lower: str) -> tuple[bool, str]:
    """
    Check if resume implies the skill.
    1. Exact key lookup in _INFER_FROM (handles LPs, soft skills, named techs).
    2. Word-level lookup in _WORD_INFER (handles dynamic responsibility phrases).

    Uses check_trigger() for all lookups:
    - Short alphanum triggers (≤5 chars) → word-boundary match (prevents
      "rds" matching "records", "eks" matching in other words)
    - Longer / punctuated triggers → substring match (specific enough)
    """
    item_lower = item.lower()

    # Pass 1: full-phrase inference
    triggers = list(_INFER_FROM.get(item_lower, []))
    for equiv in _ALIAS_LOOKUP.get(item_lower, set()):
        triggers += _INFER_FROM.get(equiv, [])
    # Also check canonical + variants from normalizer
    for variant in all_variants(item):
        triggers += _INFER_FROM.get(variant, [])
    for trigger in triggers:
        if check_trigger(trigger, resume_lower):
            return True, trigger

    # Pass 2: word-level inference (important for responsibility phrases)
    words = re.split(r"\W+", item_lower)
    for word in words:
        if len(word) < 4:
            continue
        word_triggers = _WORD_INFER.get(word, [])
        for trigger in word_triggers:
            if check_trigger(trigger, resume_lower):
                return True, trigger

    return False, ""


def _token_overlap(item: str, chunk: str) -> float:
    """
    Fraction of item's meaningful tokens present in chunk.
    Uses word-boundary checking for short tokens to avoid false positives.
    """
    chunk_lower = chunk.lower()
    toks_a = {t for t in re.split(r"\W+", item.lower())
               if len(t) > 2 and t not in _STOPS}
    if not toks_a:
        return 0.0
    found = sum(1 for t in toks_a if wcontains(t, chunk_lower))
    return found / len(toks_a)


def _expand_queries(item: str) -> list[str]:
    """
    Generate multiple query strings per JD item to improve recall.
    Returns: [raw_item, rich_query, verb_expanded_1?, verb_expanded_2?]
    """
    queries = [item]
    rich = _SEMANTIC_QUERIES.get(item, _SEMANTIC_QUERIES.get(item.lower()))
    if rich:
        queries.append(rich)

    words = item.lower().split()
    if words:
        syns = _RESP_VERB_SYNS.get(words[0], [])
        rest = " ".join(words[1:])
        for syn in syns[:2]:
            queries.append(f"{syn} {rest}".strip())

    # Add "resume evidence" templates for tech skills
    if len(words) <= 3 and not rich:
        queries.append(f"experience with {item.lower()}")
        queries.append(f"built and deployed {item.lower()}")

    return queries[:5]


# ---------------------------------------------------------------------------
# Matching engines
# ---------------------------------------------------------------------------

def _semantic_scores(
    items: list[str], chunks: list[str], model
) -> dict[str, tuple[float, str]]:
    """
    Multi-query semantic scoring.
    For each JD item, encodes multiple query variants and takes the best
    similarity across all variants × all chunks, then boosts with token overlap.
    """
    try:
        from sentence_transformers import util

        # Build flat list of all queries, tracking item boundaries
        all_queries:       list[str] = []
        item_query_slices: list[tuple[int, int]] = []

        for item in items:
            start = len(all_queries)
            all_queries.extend(_expand_queries(item))
            item_query_slices.append((start, len(all_queries)))

        query_embs = model.encode(
            all_queries, convert_to_tensor=True, show_progress_bar=False
        )
        chunk_embs = model.encode(
            chunks, convert_to_tensor=True, show_progress_bar=False
        )
        sim_mat = util.cos_sim(query_embs, chunk_embs)  # (n_queries_total, n_chunks)

        out: dict[str, tuple[float, str]] = {}
        for i, item in enumerate(items):
            s, e = item_query_slices[i]
            # max over queries → best score per chunk
            item_sim = sim_mat[s:e]             # (n_q, n_chunks)
            try:
                import torch
                max_per_chunk = torch.max(item_sim, dim=0).values
            except Exception:
                import numpy as _np
                max_per_chunk = item_sim.max(axis=0)

            best_chunk_idx = int(max_per_chunk.argmax())
            base_score     = float(max_per_chunk[best_chunk_idx])

            # Blend with token overlap for precision boost
            tok_ov = _token_overlap(item, chunks[best_chunk_idx])
            final  = min(base_score * 0.72 + tok_ov * 0.28, 0.99)

            out[item] = (final, chunks[best_chunk_idx])
        return out
    except Exception:
        return {item: (0.0, "") for item in items}


def _tfidf_scores(
    items: list[str], chunks: list[str]
) -> dict[str, tuple[float, str]]:
    """Fallback: TF-IDF cosine on expanded queries + token overlap."""
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    queries = [" ".join(_expand_queries(item)) for item in items]
    vect    = _get_tfidf()
    corpus  = queries + chunks
    try:
        tfidf_mat = vect.fit_transform(corpus).toarray()
    except Exception:
        return {item: (0.0, "") for item in items}

    n = len(items)
    query_vecs = tfidf_mat[:n]
    chunk_vecs = tfidf_mat[n:]

    if chunk_vecs.shape[0] == 0:
        return {item: (0.0, "") for item in items}

    sim_mat = cosine_similarity(query_vecs, chunk_vecs)
    out: dict[str, tuple[float, str]] = {}
    for i, item in enumerate(items):
        best_idx   = int(np.argmax(sim_mat[i]))
        base_score = float(sim_mat[i, best_idx])
        tok_ov     = _token_overlap(item, chunks[best_idx])
        combined   = min(base_score * 0.65 + tok_ov * 0.35, 0.99)
        out[item]  = (combined, chunks[best_idx])
    return out


# ---------------------------------------------------------------------------
# Category matcher
# ---------------------------------------------------------------------------

def _match_category(
    items: list[str],
    chunks: list[str],
    resume_lower: str,
    model,
) -> dict:
    if not items:
        return {"matched": [], "partial": [], "missing": [], "score": 0, "total": 0}

    exact_matched: list[dict] = []
    to_infer:      list[str]  = []

    for item in items:
        if _is_exact(item, resume_lower):
            exact_matched.append({
                "item": item, "match_type": "exact",
                "score": 1.0, "found_in": "",
            })
        else:
            to_infer.append(item)

    partial: list[dict] = []
    to_semantic: list[str] = []

    # Inference pass
    for item in to_infer:
        implied, trigger = _is_implied(item, resume_lower)
        if implied:
            partial.append({
                "item":       item,
                "match_type": "implied",
                "score":      INFER_SCORE,
                "found_in":   f"inferred from '{trigger}' in your resume",
            })
        else:
            to_semantic.append(item)

    # Semantic / TF-IDF pass
    missing: list[str] = []
    if to_semantic:
        if model:
            scores = _semantic_scores(to_semantic, chunks, model)
        else:
            scores = _tfidf_scores(to_semantic, chunks)

        for item in to_semantic:
            score, snippet = scores.get(item, (0.0, ""))
            # Additional token overlap check — if >50% of item tokens appear
            # in ANY resume chunk, treat as at least partial match
            if score < SEMANTIC_THRESHOLD:
                best_tok = max(_token_overlap(item, c) for c in chunks[:50])
                if best_tok >= 0.55:
                    score = max(score, SEMANTIC_THRESHOLD + 0.02)
                    best_c = max(chunks[:50], key=lambda c: _token_overlap(item, c))
                    snippet = best_c[:120]

            if score >= CLOSE_THRESHOLD:
                partial.append({
                    "item":       item,
                    "match_type": "close",
                    "score":      round(score, 2),
                    "found_in":   snippet[:140],
                })
            elif score >= SEMANTIC_THRESHOLD:
                partial.append({
                    "item":       item,
                    "match_type": "semantic",
                    "score":      round(score, 2),
                    "found_in":   snippet[:140],
                })
            else:
                missing.append(item)

    # Weighted score: exact=1.0, implied=0.90, close=0.78, semantic=0.50
    score_pts = len(exact_matched) * 1.0
    for p in partial:
        mt = p["match_type"]
        if mt == "implied":
            score_pts += 0.90
        elif mt == "close":
            score_pts += 0.78
        else:
            score_pts += 0.50

    total     = len(items)
    cat_score = round(score_pts / total * 100) if total else 0

    return {
        "matched": exact_matched,
        "partial": partial,
        "missing": missing,
        "score":   cat_score,
        "total":   total,
    }


# ---------------------------------------------------------------------------
# Actionable suggestions
# ---------------------------------------------------------------------------

_LP_HINTS: dict[str, str] = {
    "Ownership":
        "Add 'owned end-to-end delivery of [X]' or 'drove [project] from design to launch'",
    "Bias for Action":
        "Show speed: 'shipped [X] in [Y] weeks', 'launched MVP in [N] days'",
    "Deliver Results":
        "Quantify outcomes: 'reduced latency by X%', 'delivered [X] ahead of schedule'",
    "Think Big":
        "Add strategic work: 'led long-term vision for...', 'defined multi-year roadmap'",
    "Customer Obsession":
        "Add customer impact: 'improved retention by X%', 'reduced friction for [user type]'",
    "Dive Deep":
        "Show analysis: 'drove root-cause investigation', 'cut MTTR from X to Y'",
    "Invent & Simplify":
        "Highlight automation: 'automated [X] saving Y hours/week', 'simplified [legacy system]'",
    "Have Backbone; Disagree & Commit":
        "Show courage: 'proposed alternative approach and drove adoption'",
    "Are Right, A Lot":
        "Show judgment: 'made data-driven decisions', 'recommended [X] based on analysis'",
    "Learn & Be Curious":
        "Show growth: 'quickly learned [new tech] to deliver [X]'",
    "Hire & Develop the Best":
        "Show leadership: 'mentored [N] engineers, two of whom were promoted'",
    "Insist on Highest Standards":
        "Add quality signals: '[X]% test coverage', 'zero-downtime deployments', 'SLO'",
    "Earn Trust":
        "Show collaboration: 'built cross-team alignment on [X]', 'ran post-mortems'",
    "Frugality":
        "Show efficiency: 'reduced infra costs by X%', 'built [X] with minimal resources'",
}


def _build_suggestions(cat_results: dict) -> list[dict]:
    suggestions: list[dict] = []

    tech = cat_results.get("tech_skills", {})
    tech_missing = tech.get("missing", [])
    if tech_missing:
        count = len(tech_missing)
        top   = tech_missing[:4]
        label = f"Missing {count} tech skill{'s' if count > 1 else ''}: {', '.join(top)}"
        if count > 4:
            label += f" +{count - 4} more"
        suggestions.append({"type": "critical", "text": label})

    resp = cat_results.get("responsibilities", {})
    if resp.get("score", 100) < 55:
        n = len(resp.get("missing", []))
        if n:
            suggestions.append({
                "type": "warning",
                "text": (
                    f"{n} key responsibilit{'y' if n == 1 else 'ies'} not reflected — "
                    "rephrase bullets to match JD language"
                ),
            })

    lps = cat_results.get("amazon_lps", {})
    for lp in lps.get("missing", [])[:3]:
        hint = _LP_HINTS.get(lp, f"Show evidence of '{lp}' in your experience bullets")
        suggestions.append({"type": "lp", "text": f"LP '{lp}': {hint}"})

    for cat in ("soft_skills", "behavioral_traits"):
        res   = cat_results.get(cat, {})
        miss  = res.get("missing", [])
        score = res.get("score", 100)
        if miss and score < 50:
            suggestions.append({
                "type": "info",
                "text": f"Not visible in resume: {', '.join(miss[:3])}",
            })

    if tech.get("score", 0) >= 90 and tech.get("total", 0) > 0:
        suggestions.append({
            "type": "success",
            "text": "All required technical skills are covered — great technical alignment",
        })

    return suggestions[:8]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compare(jd_data: dict, resume_text: str) -> dict:
    """
    ATS-grade resume vs JD comparison.
    Pipeline: exact → alias → inference → multi-query semantic + token overlap.
    """
    model        = _get_st_model()
    resume_lower = resume_text.lower()
    chunks       = _split_chunks(resume_text)
    using_model  = MODEL_NAME if model else "tfidf-fallback"

    cat_results:   dict[str, dict] = {}
    weighted_score = 0.0
    total_weight   = 0.0

    for cat in ("tech_skills", "responsibilities", "amazon_lps",
                "soft_skills", "behavioral_traits"):
        items = jd_data.get(cat) or []
        if not items:
            continue
        result           = _match_category(items, chunks, resume_lower, model)
        cat_results[cat] = result
        w                = CATEGORY_WEIGHTS[cat]
        weighted_score  += result["score"] * w
        total_weight    += w

    overall = round(weighted_score / total_weight) if total_weight else 0

    if overall >= 80:
        grade, grade_color = "Strong Match", "green"
    elif overall >= 65:
        grade, grade_color = "Good Match",   "blue"
    elif overall >= 45:
        grade, grade_color = "Partial Match","yellow"
    else:
        grade, grade_color = "Weak Match",   "red"

    return {
        "overall_score": overall,
        "grade":         grade,
        "grade_color":   grade_color,
        "categories":    cat_results,
        "suggestions":   _build_suggestions(cat_results),
        "model_used":    using_model,
    }
