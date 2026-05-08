import re
import spacy
from spacy.matcher import PhraseMatcher
from collections import Counter

# ---------------------------------------------------------------------------
# Curated direct-match vocabulary
# ---------------------------------------------------------------------------

TECH_SKILLS = [
    # Languages
    "Python", "JavaScript", "TypeScript", "Java", "C", "C++", "C#", "Go",
    "Golang", "Rust", "Ruby", "PHP", "Swift", "Kotlin", "Scala", "R", "MATLAB",
    "Perl", "Bash", "Shell", "PowerShell", "Groovy", "Elixir", "Haskell",
    "Clojure", "F#", "Dart", "Lua", "Julia",
    # Web
    "HTML", "CSS", "SCSS", "SASS", "React", "React.js", "Vue", "Vue.js",
    "Angular", "Next.js", "Nuxt.js", "Svelte", "jQuery", "Bootstrap",
    "Tailwind", "Tailwind CSS", "WebSockets", "Redux", "MobX", "Zustand",
    "Storybook", "Webpack", "Vite", "Babel",
    # Backend / Frameworks
    "Node.js", "Express", "Express.js", "Django", "Flask", "FastAPI", "Spring",
    "Spring Boot", "ASP.NET", ".NET", "Laravel", "Rails", "Ruby on Rails",
    "NestJS", "Gin", "Echo", "Fiber", "Actix", "Phoenix", "Ktor",
    # Databases — full names AND common aliases
    "SQL", "NoSQL",
    "MySQL", "Postgres", "PostgreSQL",
    "SQLite", "MariaDB", "Aurora", "CockroachDB",
    "MongoDB", "Mongo",
    "Redis",
    "Elasticsearch", "Elastic", "OpenSearch",
    "Cassandra",
    "DynamoDB", "Dynamo",
    "Firebase", "Firestore",
    "Oracle", "MS SQL Server", "SQL Server",
    "BigQuery", "Snowflake", "Redshift",
    "ClickHouse", "InfluxDB", "TimescaleDB",
    "Neo4j", "ArangoDB",
    "Couchbase", "CouchDB",
    "Supabase", "PlanetScale", "Neon",
    # AWS — service names often caught as ORG by spaCy
    "AWS", "Amazon Web Services",
    "EC2", "S3", "RDS", "Lambda", "ECS", "EKS", "ECR", "Fargate",
    "CloudWatch", "CloudFormation", "CloudFront", "CloudTrail",
    "IAM", "VPC", "Route 53", "API Gateway",
    "SNS", "SQS", "SES", "Kinesis", "MSK",
    "Glue", "Athena", "EMR", "Lake Formation",
    "Step Functions", "EventBridge",
    "Elastic Beanstalk", "Lightsail", "Amplify",
    "Secrets Manager", "Parameter Store",
    "CodePipeline", "CodeBuild", "CodeDeploy",
    "Cognito", "WAF", "Shield",
    # GCP — service names
    "GCP", "Google Cloud",
    "GKE", "Cloud Run", "Cloud Functions",
    "Pub/Sub", "Cloud Storage", "Cloud SQL",
    "Dataflow", "Dataproc", "Composer",
    "BigQuery",           # already listed above
    "Vertex AI", "Cloud Build",
    "Cloud Logging", "Cloud Monitoring", "Cloud Trace",
    "Looker",
    # Azure — service names
    "Azure",
    "AKS", "Azure Functions", "Azure DevOps",
    "Azure Blob Storage", "Azure SQL",
    "CosmosDB", "Azure Cosmos DB",
    "Azure Service Bus", "Azure Event Hubs",
    "Azure AD", "Azure Active Directory",
    "Azure Monitor", "Azure Key Vault",
    # Other cloud / IaC
    "Docker", "Kubernetes", "Terraform", "Pulumi",
    "Ansible", "Chef", "Puppet", "SaltStack",
    "Helm", "Kustomize", "ArgoCD", "Flux",
    "Consul", "Vault", "Istio", "Linkerd", "Envoy",
    "Packer",
    # CI/CD & Version control
    "CI/CD", "Jenkins", "GitHub Actions", "GitLab CI",
    "CircleCI", "Travis CI", "TeamCity", "Bamboo", "Spinnaker",
    "Git", "GitHub", "GitLab", "Bitbucket",
    # Monitoring / Observability
    "Prometheus", "Grafana", "Datadog", "New Relic", "Splunk",
    "Dynatrace", "AppDynamics",
    "PagerDuty", "OpsGenie", "VictorOps",
    "Sentry", "Rollbar",
    "Jaeger", "Zipkin", "OpenTelemetry",
    "ELK Stack", "Kibana", "Logstash", "Fluentd", "Loki",
    # Message queues / streaming
    "Kafka", "Apache Kafka",
    "RabbitMQ", "ActiveMQ", "NATS", "Pulsar",
    "SQS",   # already in AWS but kept for standalone match
    # Dev tools
    "JIRA", "Confluence", "Notion",
    "Figma", "Postman", "Swagger", "OpenAPI",
    "Nx", "Turborepo",
    # Data / Analytics
    "Apache Spark", "Spark",
    "Hadoop", "Hive", "Presto", "Trino",
    "Airflow", "Prefect", "Dagster",
    "dbt", "Fivetran", "Stitch", "Airbyte",
    "Tableau", "Power BI", "Metabase", "Superset", "Redash",
    "ETL", "Data Pipeline", "Data Warehouse", "Data Lake",
    # ML / AI
    "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
    "TensorFlow", "PyTorch", "Keras", "scikit-learn",
    "XGBoost", "LightGBM", "CatBoost",
    "pandas", "NumPy", "SciPy", "Matplotlib", "Seaborn", "Plotly",
    "Hugging Face", "Transformers",
    "MLflow", "Kubeflow", "Weights & Biases", "W&B",
    "LangChain", "LlamaIndex", "LLM", "RAG",
    "Vector Database", "Pinecone", "Weaviate", "Qdrant", "Chroma",
    "MLOps",
    # Security
    "OAuth", "OAuth2", "JWT", "SSL", "TLS", "SAML", "SSO",
    "OWASP", "Penetration Testing", "Cybersecurity",
    "Zero Trust", "Okta", "Auth0", "Keycloak",
    "Snyk", "SonarQube", "Veracode",
    # Mobile
    "iOS", "Android", "React Native", "Flutter", "Xamarin",
    # Architecture & patterns
    "REST", "RESTful", "GraphQL", "gRPC", "SOAP", "WebRTC", "MQTT",
    "Microservices", "Serverless", "Event-driven", "MVC", "SOLID",
    "Design Patterns", "Object-Oriented", "OOP", "Functional Programming",
    "TDD", "BDD", "DDD", "CQRS", "Event Sourcing",
    # Methodologies
    "Agile", "Scrum", "Kanban", "DevOps", "SRE", "Platform Engineering",
    # Infra / OS
    "Linux", "Unix", "Windows Server",
    "Nginx", "Apache", "Caddy",
    "Load Balancing", "CDN", "Reverse Proxy",
    "Message Queue", "Celery",
    "Distributed Systems", "High Availability", "Fault Tolerance",
    "Caching", "Rate Limiting", "Service Mesh",
]

EXPLICIT_SOFT_SKILLS = [
    "communication", "teamwork", "leadership", "problem-solving",
    "problem solving", "analytical", "analytical skills", "detail-oriented",
    "detail oriented", "self-motivated", "self motivated", "collaborative",
    "creative", "adaptable", "time management", "critical thinking",
    "interpersonal", "fast learner", "written communication",
    "verbal communication", "presentation skills", "mentoring", "mentorship",
    "cross-functional", "stakeholder management", "project management",
    "decision making", "conflict resolution", "attention to detail",
    "team player", "proactive", "ownership", "accountability",
    "self-starter", "results-driven", "data-driven", "organized",
]

EDUCATION_PATTERNS = [
    r"bachelor'?s?\s*(?:degree)?(?:\s+in\s+[\w\s]{2,30})?",
    r"master'?s?\s*(?:degree)?(?:\s+in\s+[\w\s]{2,30})?",
    r"ph\.?d\.?(?:\s+in\s+[\w\s]{2,30})?",
    r"b\.?s\.?(?:\s+in\s+[\w\s]{2,20})?",
    r"m\.?s\.?(?:\s+in\s+[\w\s]{2,20})?",
    r"b\.?e\.?(?:\s+in\s+[\w\s]{2,20})?",
    r"b\.?tech\.?(?:\s+in\s+[\w\s]{2,20})?",
    r"m\.?tech\.?(?:\s+in\s+[\w\s]{2,20})?",
    r"associate'?s?\s*(?:degree)?",
    r"computer science",
    r"information technology",
    r"software engineering",
    r"data science",
    r"electrical engineering",
    r"mathematics",
    r"statistics",
]

EXPERIENCE_PATTERNS = [
    r"(\d+)\+?\s*(?:to|-)\s*(\d+)\s*years?",
    r"(\d+)\+?\s*years?\s+(?:of\s+)?(?:experience|exp\.?)",
    r"minimum\s+(\d+)\s*years?",
    r"at\s+least\s+(\d+)\s*years?",
]

JOB_ROLES = [
    "Software Engineer", "Software Developer", "Frontend Engineer",
    "Backend Engineer", "Full Stack Engineer", "Full Stack Developer",
    "Data Engineer", "Data Scientist", "Data Analyst", "ML Engineer",
    "Machine Learning Engineer", "AI Engineer", "DevOps Engineer",
    "Site Reliability Engineer", "Cloud Engineer", "Platform Engineer",
    "QA Engineer", "Test Engineer", "Security Engineer", "Mobile Developer",
    "iOS Developer", "Android Developer", "Product Manager", "Engineering Manager",
    "Tech Lead", "Technical Lead", "Architect", "Solutions Architect",
    "Systems Engineer", "Network Engineer", "Database Administrator", "DBA",
    "UI Developer", "UX Engineer", "Research Engineer", "Embedded Engineer",
    "Staff Engineer", "Principal Engineer", "Distinguished Engineer",
]

# ---------------------------------------------------------------------------
# Contextual soft skill detection
# Each key = canonical skill name; values = regex patterns found IN SENTENCES
# that imply this skill even without the keyword being present.
# ---------------------------------------------------------------------------

CONTEXTUAL_SOFT_SKILLS = {
    "Ownership": [
        r"takes?\s+ownership",
        r"sense\s+of\s+ownership",
        r"end[\s\-]to[\s\-]end\s+ownership",
        r"owns?\s+(?:the\s+)?(?:feature|project|product|roadmap|deliverable|service|stack)",
        r"accountab(?:le|ility)\s+for",
        r"responsible\s+for\s+(?:the\s+)?(?:full|entire|end[\s\-]to[\s\-]end)",
        r"high\s+(?:degree\s+of\s+)?ownership",
    ],
    "Problem Solving": [
        r"solv(?:e|es|ing|ed)\s+(?:complex|hard|difficult|ambiguous)?\s*(?:problems?|challenges?|issues?)",
        r"troubleshoot(?:ing)?",
        r"root[\s\-]cause\s+(?:analysis)?",
        r"break(?:ing)?\s+down\s+(?:complex|ambiguous|large|hard)\s+(?:problems?|challenges?|tasks?)",
        r"strong\s+analytical",
        r"analytical\s+(?:mind(?:set)?|thinker|approach|ability|skills?)",
        r"tackle\s+(?:hard|complex|difficult|ambiguous|challenging)",
        r"navigate\s+(?:complex|ambiguous|challenging)",
        r"think(?:ing)?\s+(?:critically|analytically)",
        r"debug(?:ging)?\s+(?:complex|hard|difficult|production)",
    ],
    "Leadership & Influence": [
        r"lead(?:s|ing)?\s+(?:a\s+)?(?:team|project|initiative|effort|squad|group|org)",
        r"driv(?:e|es|ing)\s+(?:the\s+)?(?:project|team|initiative|roadmap|vision|strategy|results)",
        r"takes?\s+(?:the\s+)?lead",
        r"mentor(?:s|ing)?\s+(?:junior|other|team|engineer)",
        r"champion(?:ing)?\s+(?:best\s+practices|change|culture)",
        r"influenc(?:e|es|ing)\s+(?:without\s+authority|across\s+teams|stakeholder)",
        r"coach(?:es|ing)?\s+(?:junior|other|team|engineer)",
        r"grow(?:ing)?\s+(?:the\s+)?(?:team|engineer|talent)",
        r"manag(?:e|es|ing)\s+(?:a\s+team|engineer|directly)",
    ],
    "Collaboration": [
        r"work(?:s|ing)?\s+(?:closely\s+)?with\s+(?:cross[\s\-]functional|multiple\s+teams|engineering|product|design)",
        r"partner(?:s|ing)?\s+with\s+(?:stakeholder|team|business|product|engineering)",
        r"cross[\s\-]functional\s+(?:team|collaboration|partner)",
        r"collaborate\s+(?:with|closely|across)",
        r"align(?:s|ing)?\s+with\s+(?:stakeholder|team|business|leadership)",
        r"work(?:s|ing)?\s+across\s+(?:team|department|function|org)",
        r"joint(?:ly)?\s+(?:build|develop|design|deliver)",
    ],
    "Communication": [
        r"present(?:s|ing)?\s+(?:to\s+)?(?:stakeholder|leadership|executive|senior|management)",
        r"articulate\s+(?:complex|technical|ideas?|concepts?|vision)",
        r"written\s+and\s+verbal",
        r"communicate\s+(?:effectively|clearly|technical|complex)",
        r"translate\s+(?:technical|complex)\s+(?:concepts?|ideas?)",
        r"write\s+(?:technical\s+)?(?:documentation|specs?|design\s+docs?|rfcs?|proposals?)",
        r"strong\s+(?:written|verbal|oral)\s+communication",
        r"present(?:ation)?\s+skills?",
        r"explain(?:s|ing)?\s+(?:technical|complex)\s+(?:concepts?|ideas?)\s+to\s+(?:non[\s\-]technical|business)",
    ],
    "Adaptability": [
        r"comfortable\s+(?:with\s+)?ambiguit",
        r"thrive\s+in\s+(?:a\s+)?(?:fast[\s\-]paced|dynamic|ambiguous|startup|high[\s\-]growth|rapidly\s+changing)",
        r"adapt(?:s|able|ability)\s+(?:to\s+)?(?:change|new|rapidly|quickly)",
        r"fast[\s\-]paced\s+environment",
        r"wear\s+(?:many|multiple)\s+hats",
        r"startup\s+(?:mindset|environment|culture|pace|experience)",
        r"ambiguous\s+(?:environment|requirement|situation|problem)",
        r"deal(?:s|ing)?\s+with\s+ambiguit",
        r"embrace\s+(?:change|ambiguity|uncertainty)",
        r"quickly\s+(?:learn|adapt|pivot|ramp)",
    ],
    "Self-Motivation & Drive": [
        r"self[\s\-](?:starter|motivated|directed|driven|sufficient)",
        r"work(?:s|ing)?\s+(?:independently|autonomously)",
        r"(?:minimal|limited|without)\s+(?:supervision|oversight|direction)",
        r"takes?\s+(?:the\s+)?initiative",
        r"bias\s+(?:for|toward)\s+action",
        r"gets?\s+things\s+done",
        r"intrinsically\s+motivated",
        r"results?[\s\-](?:oriented|driven|focused)",
        r"driven\s+(?:individual|person|engineer|professional)",
        r"go[\s\-]getter",
        r"proactive(?:ly)?",
        r"motivated\s+(?:individual|person|professional)",
        r"high\s+(?:energy|initiative|impact)",
    ],
    "Attention to Detail": [
        r"(?:eye|attention)\s+(?:for|to)\s+detail",
        r"detail[\s\-]oriented",
        r"meticulous",
        r"high[\s\-]quality\s+(?:code|work|output|deliverable)",
        r"quality[\s\-]focused",
        r"clean\s+(?:code|architecture|design|maintainable)",
        r"pride\s+(?:in|yourself)\s+(?:on\s+)?(?:quality|craft|code)",
    ],
    "Critical Thinking": [
        r"critical(?:\s+think(?:ing|er))?",
        r"data[\s\-](?:driven|informed)\s+(?:decision|approach|mindset|thinking)",
        r"(?:make|drive|inform)\s+(?:data[\s\-])?(?:driven\s+)?decisions?",
        r"strategic\s+think(?:ing|er)",
        r"first[\s\-]principles\s+think(?:ing)?",
        r"systems?\s+think(?:ing)?",
        r"think\s+(?:strategically|holistically|systematically)",
        r"question\s+(?:assumptions|the\s+status\s+quo|requirements)",
        r"second[\s\-]order\s+(?:thinking|effects?)",
    ],
    "Creativity & Innovation": [
        r"innovat(?:e|es|ing|ive|ion)",
        r"creative\s+(?:solution|approach|thinker|problem[\s\-]solver)",
        r"think(?:ing)?\s+outside\s+the\s+box",
        r"novel\s+(?:solution|approach|idea|algorithm)",
        r"push(?:ing)?\s+(?:the\s+)?(?:boundaries|envelope)",
        r"invent(?:ing)?\s+(?:new|novel|creative)",
    ],
}

# ---------------------------------------------------------------------------
# Skill-indicator patterns: what follows these phrases is a tech requirement
# ---------------------------------------------------------------------------

SKILL_INDICATORS = [
    r"experience\s+(?:with|in|using|building|designing|developing|writing|working\s+with)\s+([\w][\w\s,/+#.\-]*?)(?=\s*[.,;:()\n]|\s*\band\b|\s*\bor\b|$)",
    r"proficient\s+in\s+([\w][\w\s,/+#.\-]*?)(?=\s*[.,;:()\n]|\s*\band\b|\s*\bor\b|$)",
    r"expertise\s+in\s+([\w][\w\s,/+#.\-]*?)(?=\s*[.,;:()\n]|\s*\band\b|\s*\bor\b|$)",
    r"knowledge\s+of\s+([\w][\w\s,/+#.\-]*?)(?=\s*[.,;:()\n]|\s*\band\b|\s*\bor\b|$)",
    r"familiarity\s+with\s+([\w][\w\s,/+#.\-]*?)(?=\s*[.,;:()\n]|\s*\band\b|\s*\bor\b|$)",
    r"skilled\s+in\s+([\w][\w\s,/+#.\-]*?)(?=\s*[.,;:()\n]|\s*\band\b|\s*\bor\b|$)",
    r"background\s+in\s+([\w][\w\s,/+#.\-]*?)(?=\s*[.,;:()\n]|\s*\band\b|\s*\bor\b|$)",
    r"hands[\s\-]on\s+(?:experience\s+)?(?:with|in)\s+([\w][\w\s,/+#.\-]*?)(?=\s*[.,;:()\n]|\s*\band\b|\s*\bor\b|$)",
    r"working\s+knowledge\s+of\s+([\w][\w\s,/+#.\-]*?)(?=\s*[.,;:()\n]|\s*\band\b|\s*\bor\b|$)",
    r"strong\s+(?:foundation|background|understanding|knowledge|grasp)\s+(?:in|of)\s+([\w][\w\s,/+#.\-]*?)(?=\s*[.,;:()\n]|\s*\band\b|\s*\bor\b|$)",
    r"deep\s+(?:understanding|knowledge|expertise|experience)\s+(?:in|of|with)\s+([\w][\w\s,/+#.\-]*?)(?=\s*[.,;:()\n]|\s*\band\b|\s*\bor\b|$)",
    r"solid\s+(?:understanding|knowledge|grasp|foundation)\s+(?:in|of)\s+([\w][\w\s,/+#.\-]*?)(?=\s*[.,;:()\n]|\s*\band\b|\s*\bor\b|$)",
    r"comfortable\s+(?:working\s+)?with\s+([\w][\w\s,/+#.\-]*?)(?=\s*[.,;:()\n]|\s*\band\b|\s*\bor\b|$)",
    r"fluent\s+in\s+([\w][\w\s,/+#.\-]*?)(?=\s*[.,;:()\n]|\s*\band\b|\s*\bor\b|$)",
    r"ability\s+to\s+(?:use|work\s+with|leverage|build\s+(?:on|with)|utilize)\s+([\w][\w\s,/+#.\-]*?)(?=\s*[.,;:()\n]|\s*\band\b|\s*\bor\b|$)",
]

# Verbs that, when followed by an object, imply a technical responsibility
RESPONSIBILITY_VERBS = {
    "build", "develop", "design", "architect", "implement", "deploy",
    "maintain", "optimize", "scale", "migrate", "integrate", "create",
    "write", "test", "review", "debug", "monitor", "configure",
    "automate", "orchestrate", "containerize", "analyze", "process",
    "deliver", "ship", "refactor", "instrument", "model", "pipeline",
    "train", "fine-tune", "serve", "evaluate", "benchmark",
}

# Action verbs that typically start responsibility bullet points in JDs
_RESP_BULLET_VERBS = {
    "build", "develop", "design", "architect", "implement", "deploy",
    "maintain", "optimize", "lead", "manage", "mentor", "own", "drive",
    "deliver", "collaborate", "write", "review", "test", "monitor",
    "define", "create", "establish", "improve", "analyze", "coordinate",
    "partner", "influence", "champion", "ensure", "scale", "migrate",
    "integrate", "support", "identify", "contribute", "ship", "refactor",
    "debug", "troubleshoot", "automate", "streamline", "oversee", "launch",
    "grow", "work", "evaluate", "research", "document", "own", "drive",
    "spearhead", "establish", "shape", "influence", "own", "report",
    "present", "communicate", "translate", "evangelize", "prototype",
}

# ---------------------------------------------------------------------------
# Leadership Principles & Behavioral Signals
# Keys prefixed "LP:" are Amazon Leadership Principles.
# Keys prefixed "BT:" are general behavioral traits.
# ---------------------------------------------------------------------------

BEHAVIORAL_SIGNALS: dict[str, list[str]] = {
    # ── Amazon Leadership Principles ────────────────────────────────────────
    "LP:Customer Obsession": [
        r"customer[\s\-](?:first|centric|obsess)",
        r"customer\s+(?:needs?|experience|satisfaction|success|impact|value|outcome)",
        r"user[\s\-](?:first|centric|experience|needs?)",
        r"put(?:ting)?\s+(?:the\s+)?customer\s+first",
        r"customer[\s\-]facing",
        r"end[\s\-]user\s+(?:experience|needs?|value)",
        r"voice\s+of\s+the\s+customer",
        r"customer\s+(?:pain\s+points?|feedback|empathy)",
    ],
    "LP:Ownership": [
        r"takes?\s+(?:\w+\s+){0,2}ownership",
        r"end[\s\-]to[\s\-]end\s+ownership",
        r"sense\s+of\s+ownership",
        r"accountabl(?:e|ility)\s+for",
        r"responsible\s+for\s+(?:the\s+)?(?:full|entire|end[\s\-]to[\s\-]end)",
        r"own\s+(?:the\s+)?(?:outcome|result|delivery|roadmap|product|feature)",
        r"high\s+(?:\w+\s+)?ownership",
        r"ownership\s+(?:mindset|mentality|culture)",
        r"never\s+say\s+[\"']?that'?s?\s+not\s+my",
    ],
    "LP:Invent & Simplify": [
        r"innovat(?:e|es|ion|ive)",
        r"simplif(?:y|ies|ication)",
        r"find\s+(?:better|simpler|more\s+efficient)\s+ways?",
        r"reduce\s+complexity",
        r"novel\s+(?:approach|solution|idea|algorithm)",
        r"creative\s+(?:solution|problem[\s\-]solv)",
        r"think\s+outside\s+the\s+box",
        r"automat(?:e|ion)\s+to\s+(?:reduce|eliminate|improve)",
    ],
    "LP:Are Right, A Lot": [
        r"sound\s+(?:judgment|decisions?)",
        r"data[\s\-](?:driven|informed)\s+(?:decision|approach|mindset)",
        r"strong\s+(?:intuition|judgment|instincts?|opinions?)",
        r"calibrated\s+(?:views?|opinion)",
        r"good\s+(?:judgment|instincts?)",
        r"make\s+(?:right|correct|good)\s+(?:calls?|decisions?)\s+with\s+(?:limited|ambiguous|incomplete)",
    ],
    "LP:Learn & Be Curious": [
        r"continuously?\s+learn",
        r"growth\s+mindset",
        r"eager\s+to\s+learn",
        r"stay\s+(?:current|up[\s\-]to[\s\-]date|curious)",
        r"curious\s+(?:about|and\s+eager|individual|engineer|person|learner)",
        r"self[\s\-](?:taught|learn)",
        r"quickly\s+(?:learn|pick\s+up\s+new|ramp\s+up)",
        r"passion\s+for\s+(?:learning|technology|craft|building)",
        r"keep(?:ing)?\s+up\s+with\s+(?:trends?|latest|new\s+tech)",
    ],
    "LP:Hire & Develop the Best": [
        r"develop\s+(?:talent|engineers?|team\s+members?|people)",
        r"grow\s+(?:the\s+team|people|talent|others)",
        r"mentor(?:ing|ship)?\s+(?:junior|other|team|engineer)",
        r"raise\s+the\s+(?:bar|standard)",
        r"invest\s+in\s+(?:people|talent|your\s+team|others)",
        r"coach(?:ing|es)?\s+(?:junior|other|team|engineer)",
        r"develop\s+others",
        r"hire\s+(?:and\s+)?(?:develop|grow|train|mentor)",
        r"attract\s+(?:and\s+)?(?:develop|retain)\s+(?:talent|top)",
    ],
    "LP:Insist on Highest Standards": [
        r"high(?:est)?\s+(?:\w+\s+)?(?:standards?|quality|bar)",
        r"raise\s+(?:the\s+)?(?:engineering\s+)?bar",
        r"quality[\s\-]focused",
        r"excellence\s+in",
        r"not\s+accept(?:ing)?\s+(?:mediocrity|compromise|second[\s\-]best)",
        r"relentlessly\s+(?:improv|high|rais)",
        r"bar\s+raiser",
        r"production[\s\-](?:quality|grade|ready)",
        r"pride\s+(?:in|yourself)\s+(?:on\s+)?(?:quality|craft|code|work)",
        r"zero\s+(?:tolerance|compromise)\s+(?:for|on)\s+(?:quality|bugs?)",
    ],
    "LP:Think Big": [
        r"think\s+big",
        r"long[\s\-]term\s+(?:vision|thinking|strategy|roadmap|impact)",
        r"strategic\s+(?:vision|thinking|roadmap|direction|impact)",
        r"bold\s+(?:idea|vision|goal|bet|move)",
        r"north\s+star",
        r"transformative\s+(?:impact|change|solution)",
        r"future[\s\-](?:proof|state|vision)",
        r"10x\s+(?:thinking|growth|impact|improvement)",
        r"moonshot",
    ],
    "LP:Bias for Action": [
        r"bias\s+(?:for|toward)\s+action",
        r"move\s+(?:fast|quickly|with\s+urgency|with\s+speed)",
        r"speed\s+(?:matters?|of\s+execution|of\s+delivery)",
        r"ship\s+(?:fast|quickly|often|early)",
        r"decisiv(?:e|eness|ely)",
        r"get\s+things?\s+done",
        r"action[\s\-]oriented",
        r"minimal\s+(?:bureaucracy|process\s+overhead|red\s+tape)",
        r"(?:fast|rapid)\s+iteration",
        r"(?:fast|quick)[\s\-]paced\s+environment",
    ],
    "LP:Frugality": [
        r"do\s+more\s+with\s+less",
        r"cost[\s\-](?:effective|conscious|efficient|optimization|savings?)",
        r"resource[\s\-](?:efficient|constrained|conscious)",
        r"lean\s+(?:team|startup|approach|mindset|operations?)",
        r"efficiency\s+(?:improvement|gains?|mindset|focus)",
        r"reduce\s+(?:cost|spend|waste|overhead|toil)",
        r"frugal(?:ity)?",
    ],
    "LP:Earn Trust": [
        r"build(?:ing)?\s+trust",
        r"earn(?:ing)?\s+trust",
        r"transparency\s+(?:with|in|and\s+honesty)",
        r"honest\s+(?:feedback|communication|dialogue|conversation)",
        r"psychological\s+safety",
        r"inclusive\s+(?:culture|team|environment|workplace)",
        r"empatheti?c(?:ally)?",
        r"listen(?:ing)?\s+(?:actively|to\s+others|with\s+empathy)",
        r"vulnerable\s+(?:and|with|in\s+front\s+of)",
        r"candid\s+(?:feedback|communication|conversation)",
    ],
    "LP:Dive Deep": [
        r"dive\s+deep",
        r"root[\s\-]cause\s+(?:analysis|investigation|debugging)",
        r"detail[\s\-]oriented",
        r"attention\s+to\s+detail",
        r"meticulous",
        r"data[\s\-]driven",
        r"metrics[\s\-](?:driven|focused|obsessed)",
        r"investigate\s+(?:deeply|thoroughly|root)",
        r"understand(?:ing)?\s+(?:deeply|thoroughly|at\s+a\s+deep\s+level)",
        r"comfortable\s+(?:getting|going)\s+(?:deep|into\s+the\s+weeds)",
        r"thorough\s+(?:analysis|investigation|testing)",
    ],
    "LP:Have Backbone; Disagree & Commit": [
        r"disagree\s+and\s+commit",
        r"challenge\s+(?:assumptions?|the\s+status\s+quo|decisions?|direction)",
        r"speak\s+up",
        r"push\s+back\s+(?:constructively|when\s+needed|on\s+)",
        r"not\s+afraid\s+to\s+(?:challenge|question|voice|say\s+no)",
        r"voice\s+(?:your|their|strong)\s+(?:opinion|view|concern|disagreement)",
        r"intellectual\s+(?:courage|honesty|integrity)",
        r"constructive\s+(?:challenge|pushback|debate|tension)",
        r"respectfully\s+(?:challenge|disagree|push\s+back)",
    ],
    "LP:Deliver Results": [
        r"deliver(?:ing|s)?\s+results?",
        r"results?[\s\-](?:driven|oriented|focused)",
        r"meet(?:ing)?\s+(?:deadlines?|commitments?|goals?|targets?)",
        r"executio?n",
        r"hit(?:ting)?\s+(?:targets?|goals?|milestones?|OKRs?|KPIs?)",
        r"impact[\s\-](?:driven|focused|oriented)",
        r"outcome[\s\-](?:driven|oriented|focused)",
        r"track\s+record\s+of\s+(?:delivering|shipping|executing)",
        r"consistent(?:ly)?\s+(?:deliver|ship|execute|meet)",
    ],
    # ── General Behavioral Traits ────────────────────────────────────────────
    "BT:Growth Mindset": [
        r"growth\s+mindset",
        r"embrace\s+(?:failure|feedback|change|learning|challenges?)",
        r"view\s+(?:failure|mistakes?)\s+as\s+(?:learning|opportunity|feedback)",
        r"continuous\s+(?:improvement|learning|growth|development)",
        r"always\s+(?:improving|learning|growing|seeking\s+feedback)",
        r"not\s+afraid\s+to\s+(?:fail|make\s+mistakes?|be\s+wrong)",
    ],
    "BT:Resilience & Grit": [
        r"resilient(?:ly)?",
        r"grit",
        r"persever(?:e|ance)",
        r"bounce\s+back",
        r"thrive\s+under\s+(?:pressure|stress|adversity|constraints?)",
        r"high[\s\-]pressure\s+(?:environment|situation|deadlines?)",
        r"never\s+give\s+up",
        r"persistent(?:ly)?",
    ],
    "BT:Empathy & Inclusion": [
        r"empatheti?c",
        r"psychological\s+safety",
        r"inclusive\s+(?:culture|environment|team|workplace)",
        r"people[\s\-]first\s+(?:culture|mindset|approach)",
        r"servant\s+leader(?:ship)?",
        r"care\s+(?:deeply|about\s+(?:people|team|culture|others))",
        r"diverse\s+(?:and\s+inclusive|team|perspectives?|backgrounds?)",
        r"belong(?:ing)?\s+(?:and|for\s+all|for\s+everyone)",
    ],
    "BT:Communication & Influence": [
        r"influence\s+without\s+authority",
        r"communicate\s+(?:effectively|clearly|across\s+all\s+levels?)",
        r"translate\s+(?:technical|complex)\s+(?:concepts?|ideas?)\s+to\s+(?:non[\s\-]technical|business|stakeholder)",
        r"tailor(?:ing)?\s+(?:your\s+)?(?:message|communication)\s+(?:to|for)",
        r"executive[\s\-]level\s+communication",
        r"present(?:ing|ation)?\s+(?:to\s+)?(?:executive|c[\s\-]suite|senior\s+leadership|board)",
        r"stakeholder\s+alignment",
    ],
    "BT:Integrity & Accountability": [
        r"integrity",
        r"ethical(?:ly)?",
        r"transparent(?:ly)?",
        r"honest(?:y)?",
        r"do\s+the\s+right\s+thing",
        r"accountabl(?:e|ility)",
        r"responsible\s+(?:and\s+)?(?:transparent|accountable)",
        r"act\s+with\s+(?:integrity|honesty|ethics?)",
    ],
    "BT:Strategic Thinking": [
        r"strategic(?:ally)?",
        r"systems?\s+think(?:ing|er)",
        r"first[\s\-]principles?\s+think(?:ing)?",
        r"connect(?:ing)?\s+the\s+dots",
        r"see\s+the\s+(?:big\s+picture|forest\s+for\s+the\s+trees)",
        r"holistic(?:ally)?\s+(?:think|view|approach|understand)",
        r"long[\s\-]term\s+(?:perspective|mindset|outlook|thinker)",
    ],
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

_nlp = None

def _load_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _build_phrase_matcher(nlp, phrases):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(p.lower()) for p in phrases]
    matcher.add("MATCH", patterns)
    return matcher


def _deduplicate(items):
    items = sorted(set(items), key=len, reverse=True)
    result = []
    for item in items:
        if not any(item.lower() in kept.lower() for kept in result):
            result.append(item)
    return sorted(result, key=str.lower)


def _normalize_skill_key(s: str) -> str:
    """Collapse hyphens, spaces, '&', '/' for fuzzy deduplication."""
    return re.sub(r"[\s\-&/]+", "", s.lower())


def _extract_contextual_soft_skills(text: str) -> list[str]:
    """Scan sentences for implied soft skills using regex patterns."""
    found = []
    for skill_name, patterns in CONTEXTUAL_SOFT_SKILLS.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                found.append(skill_name)
                break  # only add each skill once
    return found


def _extract_from_skill_indicators(text: str, known_tech_lower: set) -> list[str]:
    """
    Extract tech skills from sentences like:
      'experience with Python and React'
      'proficient in distributed systems'
    Returns a list of canonical tech names (if matched) or cleaned noun phrases.
    """
    tech_lookup = {s.lower(): s for s in TECH_SKILLS}
    found = {}

    for pattern in SKILL_INDICATORS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            raw = m.group(1).strip().rstrip(".,; ")
            # Split comma-separated lists: "Python, React, Node.js"
            parts = re.split(r",\s*", raw)
            for part in parts:
                part = part.strip().rstrip(".,; ")
                if not part or len(part) < 2 or len(part.split()) > 5:
                    continue
                lower = part.lower()
                # Match against known tech (canonical casing)
                if lower in tech_lookup:
                    canonical = tech_lookup[lower]
                    found[lower] = canonical
                elif lower not in known_tech_lower:
                    # Keep short multi-word domain phrases not already captured
                    if len(part.split()) >= 2:
                        found[lower] = part.title() if part.islower() else part

    return list(found.values())


def _extract_dep_requirements(doc, known_tech_lower: set) -> list[str]:
    """
    Use spaCy dependency parser to infer technical domains from sentences like:
      'You will design and build scalable APIs'
      'Must be able to work with large-scale distributed systems'
    """
    STOP_WORDS = {
        "the", "a", "an", "our", "your", "their", "this", "that",
        "it", "we", "you", "they", "these", "those", "its",
    }
    found = {}

    for token in doc:
        if token.lemma_.lower() not in RESPONSIBILITY_VERBS:
            continue
        for child in token.children:
            if child.dep_ not in ("dobj", "attr", "nsubj"):
                continue
            span = doc[child.left_edge.i: child.right_edge.i + 1]
            phrase = span.text.strip()
            words = phrase.split()
            # Filter noise
            if not (1 <= len(words) <= 5):
                continue
            if all(w.lower() in STOP_WORDS for w in words):
                continue
            # Only keep if it contains a known tech term OR is multi-word
            has_tech = any(t in phrase.lower() for t in known_tech_lower)
            if has_tech or len(words) >= 2:
                found[phrase.lower()] = phrase

    # Deduplicate substrings
    return _deduplicate(list(found.values()))[:20]


# ---------------------------------------------------------------------------
# Responsibilities & behavioral signal extractors
# ---------------------------------------------------------------------------

def _extract_responsibilities(text: str) -> list[str]:
    """
    Extract key job responsibilities by scanning for:
    1. Bullet-point lines that start with an action verb.
    2. Sentences containing 'you will/you'll/you should + verb'.
    3. 'Responsible for ...' fragments.
    """
    responsibilities = []
    seen: set[str] = set()

    def _add(phrase: str) -> None:
        phrase = phrase.strip().rstrip(".,;:")
        # Trim to 12 words max
        phrase = " ".join(phrase.split()[:12])
        norm = phrase.lower()
        if len(norm) > 8 and norm not in seen:
            responsibilities.append(phrase)
            seen.add(norm)

    lines = text.splitlines()
    for line in lines:
        line = line.strip()
        # Strip common bullet markers
        clean = re.sub(r"^[-•*·▪◦➤→✦✓]\s*", "", line).strip()
        clean = re.sub(r"^\d+[.)]\s*", "", clean).strip()
        if len(clean) < 8:
            continue
        first = clean.split()[0].lower()
        # Rough lemma: strip common suffixes
        lemma = re.sub(r"(?:ing|ed|es|s)$", "", first)
        if first in _RESP_BULLET_VERBS or lemma in _RESP_BULLET_VERBS:
            _add(clean)

    # Sentences: "you will/you'll/you should <verb>"
    for sent in re.split(r"(?<=[.!?])\s+", text):
        sent = sent.strip()
        m = re.search(
            r"you(?:'ll|\s+will|\s+should|\s+must|\s+are\s+expected\s+to)\s+([a-z]\w*(?:\s+\w+){0,12})",
            sent, re.IGNORECASE,
        )
        if m:
            fragment = m.group(1).strip()
            first = fragment.split()[0].lower()
            lemma = re.sub(r"(?:ing|ed|es|s)$", "", first)
            if first in _RESP_BULLET_VERBS or lemma in _RESP_BULLET_VERBS:
                _add(fragment)

    # "Responsible for <phrase>" fragments
    for m in re.finditer(
        r"responsible\s+for\s+([^.,;\n]{8,80})", text, re.IGNORECASE
    ):
        _add(m.group(1))

    return responsibilities[:25]


def _detect_behavioral_signals(text: str) -> tuple[list[str], list[str]]:
    """
    Scan the JD text against BEHAVIORAL_SIGNALS patterns.
    Returns (amazon_lps, behavioral_traits) — two separate lists of matched names.
    """
    amazon_lps: list[str] = []
    behavioral_traits: list[str] = []

    for key, patterns in BEHAVIORAL_SIGNALS.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                # Strip the prefix for display
                if key.startswith("LP:"):
                    amazon_lps.append(key[3:])
                else:
                    behavioral_traits.append(key[3:])
                break  # one match per principle is enough

    return amazon_lps, behavioral_traits


# ---------------------------------------------------------------------------
# Main extraction entry point
# ---------------------------------------------------------------------------

def extract(text: str) -> dict:
    nlp = _load_nlp()
    doc = nlp(text)

    # ── 1. Direct tech skills (phrase matching) ──────────────────────────────
    tech_matcher = _build_phrase_matcher(nlp, TECH_SKILLS)
    tech_found = {}
    for _, start, end in tech_matcher(doc):
        span_text = doc[start:end].text
        canonical = next((s for s in TECH_SKILLS if s.lower() == span_text.lower()), span_text)
        tech_found[canonical.lower()] = canonical
    direct_tech = set(tech_found.values())

    # ── 2. Indicator-extracted tech ("experience with X") ────────────────────
    indicator_tech = _extract_from_skill_indicators(text, {t.lower() for t in direct_tech})
    inferred_tech_map = {}
    for item in indicator_tech:
        key = item.lower()
        if key not in tech_found:
            inferred_tech_map[key] = item
    all_tech = sorted(
        list(tech_found.values()) + list(inferred_tech_map.values()),
        key=str.lower,
    )

    # ── 3. Dependency-parsed tech domains (filtered later once NER is done) ─────
    dep_domains_raw = _extract_dep_requirements(doc, {t.lower() for t in all_tech})

    # ── 4. Explicit soft skills (phrase matching) ────────────────────────────
    soft_matcher = _build_phrase_matcher(nlp, EXPLICIT_SOFT_SKILLS)
    soft_found = {}
    for _, start, end in soft_matcher(doc):
        span_text = doc[start:end].text
        canonical = next((s for s in EXPLICIT_SOFT_SKILLS if s.lower() == span_text.lower()), span_text)
        soft_found[canonical.lower()] = canonical.title()

    # ── 5. Contextual soft skills (sentence-level regex) ─────────────────────
    contextual_soft = _extract_contextual_soft_skills(text)
    existing_norm = {_normalize_skill_key(v) for v in soft_found.values()}
    for cs in contextual_soft:
        norm = _normalize_skill_key(cs)
        # Skip if a sufficiently similar skill already exists (substring check)
        if not any(norm in ex or ex in norm for ex in existing_norm):
            soft_found[cs.lower()] = cs
            existing_norm.add(norm)
    all_soft = sorted(soft_found.values(), key=str.lower)

    # ── 6. Experience requirements ───────────────────────────────────────────
    experience = []
    for pattern in EXPERIENCE_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            experience.append(m.group(0).strip())
    experience = list(dict.fromkeys(experience))

    # ── 7. Education ─────────────────────────────────────────────────────────
    education = []
    for pattern in EDUCATION_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            raw = m.group(0).strip()
            # Reject if it crossed a newline or is suspiciously long
            if len(raw) > 3 and "\n" not in raw and len(raw) < 80:
                education.append(raw.title())
    education = _deduplicate(education)

    # ── 8. Job roles ─────────────────────────────────────────────────────────
    role_matcher = _build_phrase_matcher(nlp, JOB_ROLES)
    roles_found = {}
    for _, start, end in role_matcher(doc):
        span_text = doc[start:end].text
        canonical = next((r for r in JOB_ROLES if r.lower() == span_text.lower()), span_text)
        roles_found[canonical.lower()] = canonical
    roles = list(roles_found.values())

    # ── 9. NER — classify ORG/PRODUCT entities as tech or company mentions ──────
    # Build a full lookup of every known tech skill (lowercase → canonical)
    # so we can promote NER hits that spaCy tagged as ORG but are really tech tools.
    full_tech_lookup = {s.lower(): s for s in TECH_SKILLS}

    GENERIC_STOP = {
        "the", "and", "or", "we", "our", "team", "company", "inc", "llc",
        "corp", "ltd", "group", "he", "she", "they", "it", "its",
    }

    ner_tools = []
    for ent in doc.ents:
        if ent.label_ not in ("ORG", "PRODUCT"):
            continue
        raw = ent.text.strip()
        if not raw or len(raw) < 2 or raw.lower() in GENERIC_STOP:
            continue

        lower = raw.lower()
        if lower in full_tech_lookup:
            # Promote to tech_skills with canonical casing
            canonical = full_tech_lookup[lower]
            if lower not in tech_found:
                tech_found[lower] = canonical
        else:
            ner_tools.append(raw)

    # Rebuild all_tech to include any NER-promoted entries
    all_tech = sorted(
        list(tech_found.values()) + list(inferred_tech_map.values()),
        key=str.lower,
    )
    tech_lower_set = {t.lower() for t in all_tech}

    # Remove from ner_tools anything that ended up in tech
    ner_tools = [t for t in dict.fromkeys(ner_tools) if t.lower() not in tech_lower_set]

    # ── dep_domains: filter now that tech_lower_set is final ─────────────────
    dep_domains = [d for d in dep_domains_raw if d.lower() not in tech_lower_set]

    # ── 10. Key noun-phrase frequency ranking ────────────────────────────────
    noun_phrases = []
    seen = set()
    for chunk in doc.noun_chunks:
        phrase = chunk.text.strip()
        if "\n" in phrase or phrase.startswith("-") or phrase.startswith("•") or re.search(r"[^\w\s/+#.\-]", phrase):
            continue
        if 2 <= len(phrase.split()) <= 4 and phrase.lower() not in seen:
            if chunk.root.pos_ in ("NOUN", "PROPN") and chunk.root.is_alpha:
                # tech_lower_set is now final (after NER promotion)
                if phrase.lower() not in tech_lower_set:
                    noun_phrases.append(phrase)
                    seen.add(phrase.lower())
    freq = Counter(noun_phrases)
    top_phrases = [p for p, _ in freq.most_common(15)]

    # ── 11. Responsibilities ─────────────────────────────────────────────────
    responsibilities = _extract_responsibilities(text)

    # ── 12. Behavioral signals (Amazon LPs + general traits) ─────────────────
    amazon_lps, behavioral_traits = _detect_behavioral_signals(text)

    return {
        "tech_skills": all_tech,
        "inferred_domains": dep_domains,
        "soft_skills": all_soft,
        "responsibilities": responsibilities,
        "amazon_lps": amazon_lps,
        "behavioral_traits": behavioral_traits,
        "experience": experience,
        "education": education,
        "roles": roles,
        "other_entities": ner_tools[:10],
        "key_phrases": top_phrases,
    }
