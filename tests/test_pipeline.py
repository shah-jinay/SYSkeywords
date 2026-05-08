"""
Pipeline accuracy tests — each test targets a known failure mode.

Run with:  python -m pytest tests/ -v
           python -m pytest tests/ -v -k "word_boundary"
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from extractor import extract
from comparator import compare
from normalizer import wcontains, check_trigger, canonicalize, strip_version, all_variants


# ===========================================================================
# normalizer.py unit tests
# ===========================================================================

class TestWContains:
    """Bug 1: word-boundary safe search"""

    def test_go_not_in_mongodb(self):
        assert not wcontains("Go", "MongoDB"), \
            "'Go' should NOT match inside 'MongoDB'"

    def test_go_not_in_django(self):
        assert not wcontains("Go", "Django"), \
            "'Go' should NOT match inside 'Django'"

    def test_go_not_in_good(self):
        assert not wcontains("Go", "I'm good at coding"), \
            "'Go' should NOT match 'good'"

    def test_go_matches_standalone(self):
        assert wcontains("Go", "experience with Go and Python"), \
            "'Go' should match as standalone word"

    def test_r_not_in_javascript(self):
        assert not wcontains("R", "JavaScript developer"), \
            "'R' should NOT match inside 'JavaScript'"

    def test_r_not_in_arbitrary_text(self):
        assert not wcontains("R", "requirements analysis"), \
            "'R' should NOT match inside 'requirements'"

    def test_r_matches_standalone(self):
        assert wcontains("R", "proficient in R and Python"), \
            "'R' should match standalone"

    def test_c_not_in_react(self):
        assert not wcontains("C", "React developer"), \
            "'C' should NOT match inside 'React'"

    def test_c_matches_standalone(self):
        assert wcontains("C", "experience with C and Assembly"), \
            "'C' should match standalone"

    def test_sql_not_in_nosql(self):
        assert not wcontains("SQL", "NoSQL expert"), \
            "'SQL' should NOT match inside 'NoSQL'"

    def test_sql_matches_standalone(self):
        assert wcontains("SQL", "wrote complex SQL queries"), \
            "'SQL' should match standalone"

    def test_node_js_with_dot(self):
        assert wcontains("Node.js", "built backends in Node.js"), \
            "'Node.js' should match with the dot"

    def test_ci_cd_with_slash(self):
        assert wcontains("CI/CD", "set up CI/CD pipelines"), \
            "'CI/CD' should match with slash"

    def test_cpp_with_plus(self):
        assert wcontains("C++", "experience with C++ and Rust"), \
            "'C++' should match"

    def test_aws_not_in_draws(self):
        assert not wcontains("AWS", "he draws diagrams"), \
            "'AWS' should NOT match inside 'draws'"


class TestCheckTrigger:
    """Bug 1b: trigger matching in comparator inference"""

    def test_rds_not_in_records(self):
        assert not check_trigger("rds", "coordinate database records"), \
            "'rds' trigger should NOT match inside 'records'"

    def test_rds_matches_standalone(self):
        assert check_trigger("rds", "used amazon rds for the database"), \
            "'rds' should match as a standalone token"

    def test_long_trigger_substring_ok(self):
        assert check_trigger("root cause analysis", "conducted root cause analysis"), \
            "Long trigger phrases should use substring matching"

    def test_percent_trigger(self):
        assert check_trigger("% improvement", "achieved a 30% improvement in latency"), \
            "'% improvement' trigger should match"


class TestCanonicalize:
    """Bug 3, 5, 6: version stripping and alias normalization"""

    def test_nodejs_variants(self):
        for v in ["NodeJS", "Node JS", "nodejs", "node js"]:
            assert canonicalize(v) == "Node.js", f"'{v}' → 'Node.js'"

    def test_react_variants(self):
        for v in ["ReactJS", "React JS", "react.js"]:
            assert canonicalize(v) == "React", f"'{v}' → 'React'"

    def test_golang(self):
        assert canonicalize("Golang") == "Go"
        assert canonicalize("golang") == "Go"

    def test_postgres(self):
        assert canonicalize("Postgres") == "PostgreSQL"
        assert canonicalize("psql") == "PostgreSQL"

    def test_k8s(self):
        assert canonicalize("k8s") == "Kubernetes"
        assert canonicalize("K8s") == "Kubernetes"

    def test_cicd(self):
        assert canonicalize("CICD") == "CI/CD"
        assert canonicalize("ci cd") == "CI/CD"

    def test_strip_version_python(self):
        assert strip_version("Python 3.9") == "Python"
        assert strip_version("Python 3") == "Python"
        # "python3" has no space before "3" → NOT stripped by strip_version.
        # It's handled separately by CANONICAL_MAP ("python3" → "Python").
        assert strip_version("python3") == "python3"

    def test_product_codes_preserved_by_canonicalize(self):
        # Product codes with embedded numbers must survive canonicalize unchanged.
        # The CANONICAL_MAP has explicit entries for these so they are resolved
        # before version-stripping ever runs.
        assert canonicalize("EC2") == "EC2",      "EC2 must not be stripped to EC"
        assert canonicalize("S3") == "S3",        "S3 must not be stripped to S"
        assert canonicalize("Route 53") == "Route 53", "Route 53 is a product name"
        assert canonicalize("EKS") == "EKS"
        assert canonicalize("RDS") == "RDS"

    def test_strip_version_java(self):
        assert strip_version("Java 11") == "Java"
        assert strip_version("Java 8") == "Java"

    def test_strip_version_node(self):
        assert strip_version("Node 18") == "Node"

    def test_canonicalize_with_version(self):
        # "Python 3.9" → strip → "Python" → canonicalize → "Python"
        assert canonicalize("Python 3.9") == "Python"
        assert canonicalize("Java 11") == "Java"


class TestAllVariants:
    def test_postgres_group(self):
        variants = all_variants("PostgreSQL")
        assert "postgres" in variants
        assert "psql" in variants

    def test_kubernetes_group(self):
        variants = all_variants("Kubernetes")
        assert "k8s" in variants


# ===========================================================================
# extractor.py tests
# ===========================================================================

class TestExtractorTechSkills:
    """JD extraction must not miss skills that are clearly present."""

    def test_node_js_extracted(self):
        jd = "We need a developer with Node.js experience."
        result = extract(jd)
        skills_lower = [s.lower() for s in result["tech_skills"]]
        assert "node.js" in skills_lower, \
            f"'Node.js' not found in {result['tech_skills']}"

    def test_react_js_extracted(self):
        jd = "Experience with React.js is required."
        result = extract(jd)
        skills_lower = [s.lower() for s in result["tech_skills"]]
        assert "react" in skills_lower or "react.js" in skills_lower, \
            f"'React' not found in {result['tech_skills']}"

    def test_cicd_extracted(self):
        jd = "Must have CI/CD pipeline experience."
        result = extract(jd)
        skills_lower = [s.lower() for s in result["tech_skills"]]
        assert "ci/cd" in skills_lower, \
            f"'CI/CD' not found in {result['tech_skills']}"

    def test_go_language_extracted(self):
        jd = "Strong Go or Python development skills required."
        result = extract(jd)
        skills_lower = [s.lower() for s in result["tech_skills"]]
        assert "go" in skills_lower or "golang" in skills_lower, \
            f"'Go' not found in {result['tech_skills']}"

    def test_r_language_extracted(self):
        jd = "Proficiency in R and Python for data analysis."
        result = extract(jd)
        skills_lower = [s.lower() for s in result["tech_skills"]]
        assert "r" in skills_lower, \
            f"'R' not found in {result['tech_skills']}"

    def test_cpp_extracted(self):
        jd = "Experience with C++ is a must."
        result = extract(jd)
        skills_lower = [s.lower() for s in result["tech_skills"]]
        assert "c++" in skills_lower, \
            f"'C++' not found in {result['tech_skills']}"

    def test_postgres_canonical(self):
        jd = "Experience with Postgres and Redis required."
        result = extract(jd)
        skills_lower = [s.lower() for s in result["tech_skills"]]
        assert "postgresql" in skills_lower, \
            f"'PostgreSQL' not found (from 'Postgres') in {result['tech_skills']}"

    def test_kubernetes_from_k8s(self):
        jd = "Must know Kubernetes (k8s) for container orchestration."
        result = extract(jd)
        skills_lower = [s.lower() for s in result["tech_skills"]]
        assert "kubernetes" in skills_lower, \
            f"'Kubernetes' not found in {result['tech_skills']}"

    def test_indicator_node_js(self):
        """SKILL_INDICATORS must not stop at the dot in 'Node.js'."""
        jd = "Experience with Node.js and Express required."
        result = extract(jd)
        skills_lower = [s.lower() for s in result["tech_skills"]]
        assert "node.js" in skills_lower, \
            f"SKILL_INDICATORS stopped at dot; 'Node.js' not found in {result['tech_skills']}"

    def test_indicator_react_js(self):
        jd = "We need experience with React.js and TypeScript."
        result = extract(jd)
        skills_lower = [s.lower() for s in result["tech_skills"]]
        assert "react" in skills_lower or "react.js" in skills_lower

    def test_multi_word_phrase(self):
        jd = "Deep understanding of distributed systems and microservices architecture."
        result = extract(jd)
        all_items = (
            [s.lower() for s in result["tech_skills"]]
            + [s.lower() for s in result.get("inferred_domains", [])]
        )
        assert any("distributed" in s for s in all_items), \
            f"'Distributed systems' not found in extraction; got: {all_items}"

    def test_machine_learning_phrase(self):
        jd = "Experience with machine learning and deep learning frameworks."
        result = extract(jd)
        skills_lower = [s.lower() for s in result["tech_skills"]]
        assert "machine learning" in skills_lower, \
            f"'Machine Learning' not found in {result['tech_skills']}"

    def test_version_stripped_in_output(self):
        jd = "Requires Python 3.9 and Java 11."
        result = extract(jd)
        skills_lower = [s.lower() for s in result["tech_skills"]]
        # After canonicalize/strip_version, "python 3.9" → "python" and "java 11" → "java"
        assert "python" in skills_lower, \
            f"'Python' not found after version stripping; got {result['tech_skills']}"
        assert "java" in skills_lower, \
            f"'Java' not found after version stripping; got {result['tech_skills']}"


class TestExtractorDoesNotInvent:
    """Extractor must not fabricate keywords absent from the source text."""

    def test_no_kubernetes_if_not_mentioned(self):
        jd = "Looking for a Python developer with SQL experience."
        result = extract(jd)
        skills_lower = [s.lower() for s in result["tech_skills"]]
        assert "kubernetes" not in skills_lower

    def test_no_aws_if_not_mentioned(self):
        jd = "Django backend developer with PostgreSQL skills."
        result = extract(jd)
        skills_lower = [s.lower() for s in result["tech_skills"]]
        assert "aws" not in skills_lower


# ===========================================================================
# comparator.py tests
# ===========================================================================

class TestComparatorWordBoundary:
    """Bug 1: false positives from substring matching."""

    def test_go_not_falsely_matched(self):
        """'Go' in JD should not match a resume that only mentions MongoDB/Django."""
        jd_data = {"tech_skills": ["Go"], "responsibilities": [],
                   "amazon_lps": [], "soft_skills": [], "behavioral_traits": []}
        # This resume only has Python, Django, MongoDB — no "Go" word at all
        resume = "Built REST APIs using Django and MongoDB. Strong Python and SQL skills."
        result = compare(jd_data, resume)
        tech = result["categories"]["tech_skills"]
        matched_items = [m["item"] for m in tech["matched"]]
        partial_items = [p["item"] for p in tech["partial"]]
        # "Go" should be in missing (or semantic-only, not exact/implied)
        assert "Go" not in matched_items, \
            f"'Go' falsely matched via substring; matched={matched_items}"
        # Must not be exact-matched — may be semantic with low score, that's acceptable
        exact_matched = [m for m in tech["matched"] if m["match_type"] == "exact"]
        assert not any(m["item"] == "Go" for m in exact_matched), \
            "'Go' should not be an exact match"

    def test_r_not_falsely_matched(self):
        """'R' should not match just because common text contains the letter r."""
        jd_data = {"tech_skills": ["R"], "responsibilities": [],
                   "amazon_lps": [], "soft_skills": [], "behavioral_traits": []}
        resume = "Strong Python developer with experience in data engineering."
        result = compare(jd_data, resume)
        tech = result["categories"]["tech_skills"]
        assert "R" in tech["missing"], \
            f"'R' matched incorrectly via substring; matched={tech['matched']}"

    def test_sql_not_matched_via_nosql(self):
        """'SQL' in JD should not count as matched if resume only says 'NoSQL'."""
        jd_data = {"tech_skills": ["SQL"], "responsibilities": [],
                   "amazon_lps": [], "soft_skills": [], "behavioral_traits": []}
        resume = "Experienced with NoSQL databases like MongoDB and Cassandra."
        result = compare(jd_data, resume)
        tech = result["categories"]["tech_skills"]
        matched_items = [m["item"] for m in tech["matched"]]
        assert "SQL" not in matched_items, \
            "'SQL' should not be matched via 'NoSQL' substring"


class TestComparatorVersionNormalization:
    """Bug 3: 'Python 3.9' in JD should match 'Python' in resume."""

    def test_python_version_matches_plain(self):
        jd_data = {"tech_skills": ["Python 3.9"], "responsibilities": [],
                   "amazon_lps": [], "soft_skills": [], "behavioral_traits": []}
        resume = "5 years of Python development experience."
        result = compare(jd_data, resume)
        tech = result["categories"]["tech_skills"]
        matched_items = [m["item"] for m in tech["matched"]]
        partial_items = [p["item"] for p in tech["partial"]]
        assert "Python 3.9" in matched_items or "Python 3.9" in partial_items, \
            f"'Python 3.9' not matched against plain 'Python'; missing={tech['missing']}"

    def test_java_version_matches_plain(self):
        jd_data = {"tech_skills": ["Java 11"], "responsibilities": [],
                   "amazon_lps": [], "soft_skills": [], "behavioral_traits": []}
        resume = "Built microservices in Java using Spring Boot."
        result = compare(jd_data, resume)
        tech = result["categories"]["tech_skills"]
        matched_items = [m["item"] for m in tech["matched"]]
        partial_items = [p["item"] for p in tech["partial"]]
        assert "Java 11" in matched_items or "Java 11" in partial_items, \
            f"'Java 11' not matched against plain 'Java'; missing={tech['missing']}"


class TestComparatorAliases:
    """Aliases and alternate spellings must match on both sides."""

    def test_postgres_matches_postgresql(self):
        jd_data = {"tech_skills": ["PostgreSQL"], "responsibilities": [],
                   "amazon_lps": [], "soft_skills": [], "behavioral_traits": []}
        resume = "Managed Postgres databases and wrote complex queries."
        result = compare(jd_data, resume)
        tech = result["categories"]["tech_skills"]
        matched_items = [m["item"] for m in tech["matched"]]
        assert "PostgreSQL" in matched_items, \
            f"'PostgreSQL' not matched via 'Postgres' alias; missing={tech['missing']}"

    def test_kubernetes_from_eks(self):
        jd_data = {"tech_skills": ["Kubernetes"], "responsibilities": [],
                   "amazon_lps": [], "soft_skills": [], "behavioral_traits": []}
        resume = "Deployed microservices on EKS clusters."
        result = compare(jd_data, resume)
        tech = result["categories"]["tech_skills"]
        partial_items = [p["item"] for p in tech["partial"]]
        assert "Kubernetes" in partial_items, \
            f"'Kubernetes' not implied from 'EKS'; missing={tech['missing']}"

    def test_cicd_from_github_actions(self):
        jd_data = {"tech_skills": ["CI/CD"], "responsibilities": [],
                   "amazon_lps": [], "soft_skills": [], "behavioral_traits": []}
        resume = "Set up GitHub Actions pipelines for automated testing."
        result = compare(jd_data, resume)
        tech = result["categories"]["tech_skills"]
        partial_items = [p["item"] for p in tech["partial"]]
        matched_items = [m["item"] for m in tech["matched"]]
        assert "CI/CD" in partial_items or "CI/CD" in matched_items, \
            f"'CI/CD' not matched/implied from 'GitHub Actions'; missing={tech['missing']}"

    def test_python_from_pyspark(self):
        jd_data = {"tech_skills": ["Python"], "responsibilities": [],
                   "amazon_lps": [], "soft_skills": [], "behavioral_traits": []}
        resume = "Built data pipelines with PySpark and Airflow."
        result = compare(jd_data, resume)
        tech = result["categories"]["tech_skills"]
        partial_items = [p["item"] for p in tech["partial"]]
        matched_items = [m["item"] for m in tech["matched"]]
        assert "Python" in partial_items or "Python" in matched_items, \
            f"'Python' not inferred from 'PySpark'; missing={tech['missing']}"


class TestComparatorLPInference:
    """Amazon LPs must be inferred from resume evidence correctly."""

    def test_deliver_results_from_ahead_of_schedule(self):
        jd_data = {"tech_skills": [], "responsibilities": [],
                   "amazon_lps": ["Deliver Results"],
                   "soft_skills": [], "behavioral_traits": []}
        resume = "Delivered the migration 2 weeks ahead of schedule."
        result = compare(jd_data, resume)
        lps = result["categories"]["amazon_lps"]
        partial = [p["item"] for p in lps["partial"]]
        assert "Deliver Results" in partial, \
            f"'Deliver Results' not implied from 'ahead of schedule'; missing={lps['missing']}"

    def test_dive_deep_from_root_cause(self):
        jd_data = {"tech_skills": [], "responsibilities": [],
                   "amazon_lps": ["Dive Deep"],
                   "soft_skills": [], "behavioral_traits": []}
        resume = "Drove root-cause analysis on production incidents."
        result = compare(jd_data, resume)
        lps = result["categories"]["amazon_lps"]
        partial = [p["item"] for p in lps["partial"]]
        assert "Dive Deep" in partial, \
            f"'Dive Deep' not implied from 'root-cause'; missing={lps['missing']}"

    def test_invent_simplify_from_proactively(self):
        jd_data = {"tech_skills": [], "responsibilities": [],
                   "amazon_lps": ["Invent & Simplify"],
                   "soft_skills": [], "behavioral_traits": []}
        resume = "Proactively built a self-healing alerting framework saving 8h/week."
        result = compare(jd_data, resume)
        lps = result["categories"]["amazon_lps"]
        partial = [p["item"] for p in lps["partial"]]
        assert "Invent & Simplify" in partial, \
            f"'Invent & Simplify' not implied; missing={lps['missing']}"


class TestComparatorFalseNegatives:
    """Real resume evidence must not be silently dropped as missing."""

    def test_communication_from_stakeholder(self):
        jd_data = {"tech_skills": [], "responsibilities": [],
                   "amazon_lps": [], "soft_skills": ["communication"],
                   "behavioral_traits": []}
        resume = "Presented weekly status updates to senior stakeholders."
        result = compare(jd_data, resume)
        soft = result["categories"]["soft_skills"]
        partial = [p["item"] for p in soft["partial"]]
        matched = [m["item"] for m in soft["matched"]]
        assert "communication" in partial or "communication" in matched, \
            f"'communication' should be inferred from stakeholder context; missing={soft['missing']}"

    def test_teamwork_from_collaborated(self):
        jd_data = {"tech_skills": [], "responsibilities": [],
                   "amazon_lps": [], "soft_skills": ["teamwork"],
                   "behavioral_traits": []}
        resume = "Collaborated with product, design, and data teams to ship features."
        result = compare(jd_data, resume)
        soft = result["categories"]["soft_skills"]
        partial = [p["item"] for p in soft["partial"]]
        matched = [m["item"] for m in soft["matched"]]
        assert "teamwork" in partial or "teamwork" in matched, \
            f"'teamwork' should be inferred from 'collaborated'; missing={soft['missing']}"
