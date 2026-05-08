"""
Shared normalization and safe-search utilities.
Both extractor.py and comparator.py import from here so they use identical logic.
"""
import re

# ---------------------------------------------------------------------------
# Word-boundary safe search
# ---------------------------------------------------------------------------

def wcontains(needle: str, haystack: str) -> bool:
    """
    Return True if `needle` appears as a whole word (or term) in `haystack`.

    Handles:
    - Single-letter languages: "R" won't match inside "JavaScript"
    - Short terms: "Go" won't match inside "MongoDB" or "Django"
    - Punctuated terms: "Node.js", "C++", "CI/CD" matched correctly
    - Case-insensitive

    Uses negative lookahead/lookbehind on alphanumeric chars so the boundary
    check works for ASCII tech terms without relying on \b (which breaks on
    non-alphanumeric characters like "+" in "C++").
    """
    if not needle or not haystack:
        return False
    pattern = r'(?<![A-Za-z0-9])' + re.escape(needle) + r'(?![A-Za-z0-9])'
    return bool(re.search(pattern, haystack, re.IGNORECASE))


def check_trigger(trigger: str, text: str) -> bool:
    """
    Check whether a trigger phrase appears in text.

    Short single-word triggers (≤5 chars, purely alphanumeric) use word-boundary
    matching to prevent "rds" from matching inside "records".
    Longer triggers and phrases with punctuation use plain substring matching —
    they are specific enough that false positives are unlikely.
    """
    if not trigger or not text:
        return False
    # Multi-word or contains non-alphanumeric: specific enough → substring ok
    if ' ' in trigger or re.search(r'[^A-Za-z0-9]', trigger):
        return trigger.lower() in text.lower()
    # Short alphanum single word → word-boundary check
    if len(trigger) <= 5:
        return wcontains(trigger, text)
    # Longer single word → substring is fine (specific enough)
    return trigger.lower() in text.lower()


# ---------------------------------------------------------------------------
# Canonical alias map
# Every key is a lowercase variation; value is the preferred display form.
# ---------------------------------------------------------------------------

CANONICAL_MAP: dict[str, str] = {
    # JavaScript ecosystem
    "javascript":      "JavaScript",
    "js":              "JavaScript",
    "ecmascript":      "JavaScript",
    "es6":             "JavaScript",
    "es2015":          "JavaScript",
    "typescript":      "TypeScript",
    "ts":              "TypeScript",
    "node.js":         "Node.js",
    "nodejs":          "Node.js",
    "node js":         "Node.js",
    "react.js":        "React",
    "reactjs":         "React",
    "react js":        "React",
    "vue.js":          "Vue.js",
    "vuejs":           "Vue.js",
    "vue js":          "Vue.js",
    "next.js":         "Next.js",
    "nextjs":          "Next.js",
    "next js":         "Next.js",
    "angularjs":       "Angular",
    "angular js":      "Angular",
    "nuxt.js":         "Nuxt.js",
    "nuxtjs":          "Nuxt.js",

    # Python
    "python3":         "Python",
    "python 3":        "Python",
    "python2":         "Python",
    "python 2":        "Python",

    # Java
    "java 11":         "Java",
    "java 8":          "Java",
    "java 17":         "Java",
    "java se":         "Java",
    "java ee":         "Java",

    # Go
    "golang":          "Go",
    "go lang":         "Go",

    # C family
    "c/c++":           "C++",
    "c / c++":         "C++",
    "objective-c":     "Objective-C",

    # Databases
    "postgres":        "PostgreSQL",
    "psql":            "PostgreSQL",
    "pg":              "PostgreSQL",
    "mongo":           "MongoDB",
    "mysql 8":         "MySQL",
    "mssql":           "SQL Server",
    "ms sql":          "SQL Server",
    "ms sql server":   "SQL Server",
    "dynamo":          "DynamoDB",
    "dynamo db":       "DynamoDB",
    "cosmos db":       "CosmosDB",
    "elastic":         "Elasticsearch",
    "opensearch":      "Elasticsearch",

    # Cloud
    "amazon web services": "AWS",
    "aws cloud":       "AWS",
    "google cloud":    "GCP",
    "google cloud platform": "GCP",
    "gcp cloud":       "GCP",
    "microsoft azure": "Azure",
    "azure cloud":     "Azure",

    # AWS services (normalize variants)
    "amazon s3":       "S3",
    "amazon ec2":      "EC2",
    "amazon rds":      "RDS",
    "amazon sqs":      "SQS",
    "amazon sns":      "SNS",
    "amazon eks":      "EKS",
    "amazon ecs":      "ECS",
    "amazon lambda":   "Lambda",
    "aws lambda":      "Lambda",

    # Kubernetes / containers
    "k8s":             "Kubernetes",
    "kube":            "Kubernetes",
    "kubectl":         "Kubernetes",
    "docker compose":  "Docker",
    "docker-compose":  "Docker",
    "containerization": "Docker",

    # CI/CD
    "cicd":            "CI/CD",
    "ci cd":           "CI/CD",
    "ci/cd pipeline":  "CI/CD",
    "continuous integration": "CI/CD",
    "continuous delivery":    "CI/CD",
    "continuous deployment":  "CI/CD",
    "github action":   "GitHub Actions",
    "gha":             "GitHub Actions",
    "gitlab-ci":       "GitLab CI",

    # Messaging / streaming
    "apache kafka":    "Kafka",
    "apache spark":    "Spark",
    "pyspark":         "Spark",
    "rabbitmq":        "RabbitMQ",

    # ML / AI
    "ml":              "Machine Learning",
    "machine learning": "Machine Learning",
    "deep learning":   "Deep Learning",
    "dl":              "Deep Learning",
    "nlp":             "NLP",
    "natural language processing": "NLP",
    "computer vision": "Computer Vision",
    "cv":              "Computer Vision",
    "llm":             "LLM",
    "large language model": "LLM",
    "rag":             "RAG",
    "retrieval augmented generation": "RAG",
    "scikit learn":    "scikit-learn",
    "sklearn":         "scikit-learn",

    # AWS service product names with numbers — must NOT be stripped
    "route 53":        "Route 53",
    "ec2":             "EC2",
    "s3":              "S3",
    "rds":             "RDS",
    "ecs":             "ECS",
    "eks":             "EKS",
    "ecr":             "ECR",

    # IaC / infra
    "infrastructure as code": "Terraform",
    "iac":             "Terraform",

    # Protocols / patterns
    "restful":         "REST",
    "rest api":        "REST",
    "restful api":     "REST",
    "http api":        "REST",
    "graphql api":     "GraphQL",
    "grpc":            "gRPC",
    "oop":             "Object-Oriented",
    "object oriented": "Object-Oriented",
    "object-oriented programming": "Object-Oriented",

    # Methodologies
    "agile methodology": "Agile",
    "scrum methodology": "Scrum",

    # Testing
    "tdd":             "TDD",
    "test driven development": "TDD",
    "bdd":             "BDD",
    "behavior driven development": "BDD",
}

# Build reverse lookup: canonical → set of all aliases (for comparator)
# Key: lowercase canonical → set of all lowercase variants
ALIAS_GROUPS: dict[str, set[str]] = {}
for _variant, _canonical in CANONICAL_MAP.items():
    _key = _canonical.lower()
    if _key not in ALIAS_GROUPS:
        ALIAS_GROUPS[_key] = {_key}
    ALIAS_GROUPS[_key].add(_variant.lower())
    ALIAS_GROUPS[_key].add(_canonical.lower())

# Also allow lookup by any variant
VARIANT_TO_CANONICAL: dict[str, str] = {
    k.lower(): v for k, v in CANONICAL_MAP.items()
}
VARIANT_TO_CANONICAL.update({v.lower(): v for v in CANONICAL_MAP.values()})


# ---------------------------------------------------------------------------
# Normalization functions
# ---------------------------------------------------------------------------

# Version suffix: requires at least one space before the number so that
# product codes like "EC2", "S3", "Log4j" are NOT stripped.
# "Python 3.9" → strip ✓  |  "EC2" → keep ✓  |  "Java 11" → strip ✓
_VERSION_RE = re.compile(
    r'\s+(?:v\.?|version\s*)?\d+(?:\.\d+)*\s*$', re.IGNORECASE
)

def strip_version(term: str) -> str:
    """
    Remove trailing space-separated version numbers from a tech term.
    "Python 3.9" → "Python", "Java 11" → "Java"
    Does NOT touch product codes like "EC2", "S3", "Route 53".
    """
    stripped = _VERSION_RE.sub('', term).strip()
    return stripped if stripped else term


def canonicalize(term: str) -> str:
    """
    Return the canonical display form of a tech term.

    Order:
      1. Direct map lookup  (handles "postgres" → "PostgreSQL", "ec2" → "EC2", etc.)
      2. Version-stripped lookup  ("Python 3.9" → strip → "Python" → lookup)
      3. Return version-stripped form if stripping changed something
      4. Return original term

    Product codes like "EC2", "S3", "Route 53" are in the CANONICAL_MAP explicitly
    so they are returned as-is before version stripping ever runs.
    """
    t = term.strip()
    lower = t.lower()

    # Pass 1: direct lookup (including product codes and common aliases)
    if lower in VARIANT_TO_CANONICAL:
        return VARIANT_TO_CANONICAL[lower]

    # Pass 2: version-stripped lookup
    stripped = strip_version(t).strip()
    stripped_lower = stripped.lower()
    if stripped_lower != lower:
        if stripped_lower in VARIANT_TO_CANONICAL:
            return VARIANT_TO_CANONICAL[stripped_lower]
        # Strip was meaningful (e.g. "Python 3.9" → "Python") but no map entry
        if stripped:
            return stripped

    return t


def all_variants(canonical_or_variant: str) -> set[str]:
    """
    Return the full set of lowercase aliases for a term (including itself).
    Used by the comparator to check whether any variant appears in the resume.
    """
    lower = canonical_or_variant.lower()
    # First canonicalize
    canon_lower = canonicalize(lower).lower()
    return ALIAS_GROUPS.get(canon_lower, {lower, canon_lower})


def normalize_list(terms: list[str]) -> list[str]:
    """
    Canonicalize each term, deduplicate preserving longest form.
    Safe to run on both JD-side and resume-side term lists.
    """
    seen_canon: dict[str, str] = {}  # canonical_lower → best surface form
    for t in terms:
        canon = canonicalize(t)
        key = canon.lower()
        # Keep the longer/more specific surface form
        if key not in seen_canon or len(t) > len(seen_canon[key]):
            seen_canon[key] = canon
    return sorted(seen_canon.values(), key=str.lower)
