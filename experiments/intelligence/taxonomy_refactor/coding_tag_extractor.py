"""
Deterministic Coding Tag Extractor for Screenshot Intelligence System
======================================================================
Extracts fine-grained, evidence-supported coding and technology tags directly
from OCR text and metadata notes without hallucinating unsupported terms.

Supported Tag Schema:
- General Coding: Coding, Software Development, Programming
- Languages: Python, Java, C++, JavaScript, TypeScript, HTML, CSS, SQL, PHP
- Tools & Platforms: Git, GitHub, VS Code, Terminal, IDE, Repository, Debugger, Traceback, LeetCode, Kaggle, Jupyter, Docker, Kubernetes, API
- Topics: Algorithm, Machine Learning, Web Development
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set


TAG_RULES: Dict[str, List[str]] = {
    "Coding": [
        r"\b(def|class|import|return|function|const|let|var|public\s+static|void)\b",
        r"\b(vs\s*code|vscode|github|git|terminal|pip|npm|syntaxerror|traceback|compiler|code|programming|script|python|java|javascript|cpp|c\+\+|html|css|sql)\b",
        r"\b(leetcode|kaggle|jupyter|ipynb|array|pointer|algorithm|recursion|tree|graph|tutorial)\b",
    ],
    "Software Development": [
        r"\b(vs\s*code|vscode|github|git|commit|pull\s+request|repository|repo|build|compile|deploy|npm|pip|docker|api)\b",
        r"\b(src|main\.py|index\.html|app\.js|package\.json|requirements\.txt)\b",
    ],
    "Programming": [
        r"\b(python|javascript|typescript|java|c\+\+|cpp|c#|php|html|css|sql|coding|programmer|code|script)\b",
    ],
    "Python": [
        r"\b(python|pytest|pip|pandas|numpy|scikit-learn|matplotlib|keras|tensorflow|pytorch|def\s+[a-zA-Z0-9_]+|import\s+[a-zA-Z0-9_]+)\b",
    ],
    "Java": [
        r"\b(java|javac|maven|gradle|public\s+class|public\s+static\s+void\s+main)\b",
    ],
    "C++": [
        r"\b(c\+\+|cpp|g\+\+|#include\s+<|std::|cout|cin|vector<)\b",
    ],
    "JavaScript": [
        r"\b(javascript|js|node|express|react|vue|angular|npm|const\s+[a-zA-Z0-9_]+|let\s+[a-zA-Z0-9_]+|console\.log)\b",
    ],
    "TypeScript": [
        r"\b(typescript|ts|tsconfig|interface\s+[a-zA-Z0-9_]+|type\s+[a-zA-Z0-9_]+)\b",
    ],
    "HTML": [
        r"\b(html|doctype|<div>|<span>|<body>|<head>|<script>|<link)\b",
    ],
    "CSS": [
        r"\b(css|stylesheet|margin|padding|display:\s*flex|grid|color:)\b",
    ],
    "SQL": [
        r"\b(sql|mysql|postgresql|sqlite|select\s+.*\s+from|insert\s+into|database|table|query)\b",
    ],
    "PHP": [
        r"\b(php|<\?php|\$_get|\$_post|\$_session)\b",
    ],
    "Git": [
        r"\b(git|git\s+commit|git\s+push|git\s+pull|git\s+fetch|git\s+status|git\s+branch|git\s+checkout)\b",
    ],
    "GitHub": [
        r"\b(github|github\.com|pull\s+request|gh-pages|fork|clone)\b",
    ],
    "VS Code": [
        r"\b(vs\s*code|vscode|visual\s+studio\s+code)\b",
    ],
    "Terminal": [
        r"\b(terminal|cmd|powershell|bash|zsh|git\s+bash|mingw|prompt|ps\s+c:)\b",
    ],
    "IDE": [
        r"\b(ide|vs\s*code|vscode|intellij|pycharm|android\s+studio|eclipse|xcode)\b",
    ],
    "Repository": [
        r"\b(repository|repo|github|gitlab|bitbucket)\b",
    ],
    "Traceback": [
        r"\b(traceback|stack\s+trace|exception|syntaxerror|nameerror|typeerror|valueerror|indexerror)\b",
    ],
    "LeetCode": [
        r"\b(leetcode|leetcode\.com|two\s+sum|add\s+two\s+numbers|longest\s+substring|problem\s+list)\b",
    ],
    "Kaggle": [
        r"\b(kaggle|kaggle\.com|kaggle\s+notebook)\b",
    ],
    "Algorithm": [
        r"\b(algorithm|bfs|dfs|binary\s+search|sorting|bubble\s+sort|merge\s+sort|quick\s+sort|dijkstra|maze)\b",
    ],
    "Machine Learning": [
        r"\b(machine\s+learning|linear\s+regression|scikit-learn|pandas|numpy|neural\s+network|deep\s+learning|dataset)\b",
    ],
}


def extract_deterministic_tags(text: str, notes: str = "") -> List[str]:
    """
    Extracts all evidence-supported tags from text and notes.

    Args:
        text: Raw or normalized OCR text
        notes: Sample metadata notes

    Returns:
        List of unique tag strings
    """
    combined_text = (text + " " + notes).lower()
    matched_tags: List[str] = []

    for tag, patterns in TAG_RULES.items():
        if any(re.search(pat, combined_text, flags=re.IGNORECASE) for pat in patterns):
            matched_tags.append(tag)

    # Contextual inference for Software Development
    if ("VS Code" in matched_tags or "Git" in matched_tags or "GitHub" in matched_tags) and "Coding" in matched_tags:
        if "Software Development" not in matched_tags:
            matched_tags.append("Software Development")

    return list(dict.fromkeys(matched_tags))
