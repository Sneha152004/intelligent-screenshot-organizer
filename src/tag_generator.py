"""
Deterministic Tag Generator Module
==================================
Extracts evidence-backed, normalized descriptive tags from OCR text and contextual metadata.

Tag Families:
1. Domain / Activity: Coding, Programming, Software Development, Shopping, Learning, Gaming, Entertainment, Networking, Financial
2. Entity / Application:
   - Languages: Python, JavaScript, TypeScript, Java, C++, C#, HTML, CSS, SQL, PHP
   - Developer Tools: VS Code, Git, GitHub, Terminal, IDE, LeetCode, Kaggle, Docker, Jupyter
   - Platforms & Merchants: Amazon, Flipkart, Myntra, Swiggy, Zomato, Netflix, YouTube, Spotify
   - Payment Systems: UPI, Google Pay, PhonePe, Paytm, PayPal
3. Content Type:
   - Receipt, Invoice, Order, Ticket, Tutorial, Form, Resume, Job Posting, Chat, Email, Settings, Dashboard, Error Log

Guiding Principles:
- Deterministic, explainable regex patterns with word boundaries.
- Contextual multi-word phrase matching to eliminate substring collisions (e.g. "in order to", "pay attention").
- No hallucination: return empty or smaller tag list when evidence is insufficient.
- Preserves top-level canonical category semantics (Coding is a tag, NOT a top-level category).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple


# ── Tag Rule Definitions with Contextual Patterns ─────────────────────────────

TAG_FAMILIES: Tuple[str, ...] = ("Domain/Activity", "Entity/Application", "Content Type")

# Domain / Activity Rules
DOMAIN_RULES: Dict[str, List[str]] = {
    "Coding": [
        r"\b(def\s+[a-zA-Z_]\w*\s*\(|class\s+[a-zA-Z_]\w*\s*[:{(\[<]|class\s+[a-zA-Z_]\w*\s+(?:extends|implements)|import\s+[a-zA-Z_]\w*|return\s+[\w\(\[\{\'\"])",
        r"\b(function\s+[a-zA-Z_]\w*|(const|let|var)\s+[a-zA-Z_]\w*\s*=|console\.(log|error|warn|info))\b",
        r"\b(public\s+static\s+void|#include\s*<[a-z.]+>|namespace\s+[a-zA-Z_]|std::)\b",
        r"\b(syntaxerror|traceback|indentationerror|nullpointerexception|referenceerror)\b",
        r"\b(git\s+(commit|push|pull|status|branch|checkout|diff|log|merge))\b",
        r"\b(vs\s*code|vscode|pycharm|intellij|jupyter\s+notebook|leetcode|kaggle\s+notebook)\b",
        r"\b(npm\s+(install|run|start|test)|pip\s+install|cargo\s+build|mvn\s+clean|pytest|unittest)\b",
        r"\b(select\s+\*\s+from\s+[a-zA-Z_]\w*|select\s+[\w,\s]+\s+from\s+[a-zA-Z_]\w*\s+(?:where|group\s+by|order\s+by|limit|inner\s+join|left\s+join|;)|insert\s+into\s+[a-zA-Z_]\w*\s*(?:values|\())\b",
    ],
    "Programming": [
        r"\b(python|javascript|typescript|java|c\+\+|cpp|c#|golang|rust|ruby|php|sql|html|css)\b",
        r"\b(programming\s+(language|course|assignment|tutorial|exercise|problem))\b",
        r"\b(algorithm|data\s+structure|recursion|binary\s+tree|dynamic\s+programming|time\s+complexity)\b",
        r"\b(source\s+code|code\s+snippet|compile|debugger|breakpoint|compiler)\b",
    ],
    "Software Development": [
        r"\b(github|gitlab|bitbucket)\.com\b",
        r"\b(pull\s+request|merge\s+request|commit\s+[0-9a-f]{7,40}|git\s+repo(sitory)?)\b",
        r"\b(software\s+(engineer|engineering|developer|development))\b",
        r"\b(ci\/cd|pipeline|dockerfile|docker\s+compose|kubernetes|k8s|deploy(ment)?\s+build)\b",
        r"\b(api\s+(endpoint|documentation|request|response|key))\b",
        r"\b(package\.json|requirements\.txt|cargo\.toml|pom\.xml|tsconfig\.json)\b",
    ],
    "Shopping": [
        r"\b(add\s+to\s+(cart|bag)|buy\s+now|proceed\s+to\s+checkout)\b",
        r"\b(order\s+(summary|total|placed|confirmed|details|history|status|id|#))\b",
        r"\b(product\s+(details|description|specifications|reviews|rating))\b",
        r"\b(delivery\s+(address|estimate|by|status|tracking)|shipped\s+with|out\s+for\s+delivery)\b",
        r"\b(free\s+shipping|in\s+stock|out\s+of\s+stock|return\s+policy|item\s+subtotal)\b",
        r"\b(discount|coupon\s+code|promo\s+code|save\s+\$?\d+)\b",
    ],
    "Learning": [
        r"\b(tutorial|lecture\s+(slides?|notes?)|course\s+(material|syllabus|overview))\b",
        r"\b(study\s+(guide|notes?|material)|exam\s+(preparation|schedule|revision|questions?))\b",
        r"\b(quiz|assignment\s+#?\d*|homework|textbook|chapter\s+\d+|curriculum)\b",
        r"\b(coursera|udemy|edx|khan\s+academy|nptel|mit\s+ocw|documentation\s+tutorial)\b",
    ],
    "Gaming": [
        r"\b(gameplay|leaderboard|high\s+score|level\s+\d+|quest\s+completed)\b",
        r"\b(fps|steam|playstation|xbox|nintendo|epic\s+games|game\s+over|victory\s+royale)\b",
    ],
    "Entertainment": [
        r"\b(watch\s+(now|trailer)|now\s+playing|season\s+\d+\s+episode\s+\d+)\b",
        r"\b(netflix|prime\s+video|disney\+|hulu|spotify|apple\s+music|youtube\s+(channel|video))\b",
        r"\b(playlist|soundtrack|album|artist|podcast\s+episode)\b",
    ],
    "Networking": [
        r"\b(wi-?fi\s+(network|settings?|password|connected|connection))\b",
        r"\b(ssid|ip\s+address|mac\s+address|default\s+gateway|network\s+gateway|subnet\s+mask|dns\s+server)\b",
        r"\b(bluetooth\s+(paired|pairing|devices?)|mobile\s+hotspot|airplane\s+mode)\b",
        r"\b(cellular\s+data|sim\s+card|network\s+(settings?|status|adapter))\b",
    ],
    "Financial": [
        r"\b(bank\s+account|account\s+balance|credit\s+card|debit\s+card|payment\s+gateway)\b",
        r"\b(available\s+balance|ledger\s+balance|statement\s+period)\b",
        r"\b(fund\s+transfer|neft|rtgs|imps|wire\s+transfer)\b",
    ],
}

# Entity / Application Rules
ENTITY_RULES: Dict[str, List[str]] = {
    # Programming Languages
    "Python": [
        r"\b(python|python3|pip\s+install|pytest|pandas|numpy|scikit-learn|matplotlib|pytorch|torch|tensorflow|keras|fastapi|flask|django)\b",
        r"\b(def\s+[a-zA-Z_]\w*|import\s+[a-zA-Z_]\w*|__init__\.py|\.py\b)\b",
    ],
    "JavaScript": [
        r"\b(javascript|node\.?js|react\.?js|react\s+(native|app|component|dom|hooks?)|vue\.?js|angular|express\.js|express\s+(server|app|router)|require\(['\"]express['\"]\)|from\s+['\"]express['\"]|npm|yarn|package\.json)\b|\bexpress\s*\(",
        r"\b(const|let|var)\s+[a-zA-Z_]\w*\s*=",
        r"\bconsole\.(log|error|warn|info)\b",
    ],
    "TypeScript": [
        r"\b(typescript|tsconfig\.json|interface\s+[A-Z]\w*|type\s+[A-Z]\w*\s*=|\.tsx?\b)\b",
    ],
    "Java": [
        r"\b(java|javac|maven|gradle|spring\s+boot|public\s+class\s+[A-Z]\w*|public\s+static\s+void\s+main)\b",
    ],
    "C++": [
        r"\b(c\+\+|cpp|#include\s*<[a-z_]+>|std::cout|std::vector|std::string)\b",
    ],
    "HTML": [
        r"\b(html5?|<!doctype\s+html>|<div\b|<span\b|<a\s+href=|<form\b|<table\b|<button\b)\b",
    ],
    "CSS": [
        r"\b(css3?|stylesheet|display:\s*(flex|grid)|margin:\s*\d|padding:\s*\d|font-family:|border-radius:)\b",
    ],
    "SQL": [
        r"\b(sql|mysql|postgresql|sqlite3?|select\s+[\w*,\s]+\s+from|insert\s+into\s+\w+|create\s+table\s+\w+)\b",
    ],
    "PHP": [
        r"\b(php|<\?php|\$_get|\$_post|\$_session|\$this->)\b",
    ],
    # Developer Tools
    "VS Code": [
        r"\b(vs\s*code|vscode|visual\s+studio\s+code)\b",
    ],
    "Git": [
        r"\b(git\s+(commit|push|pull|status|branch|checkout|diff|log|merge|fetch|clone|add|init))\b",
        r"\b(git\s+repo(sitory)?|\.gitignore)\b",
    ],
    "GitHub": [
        r"\b(github|github\.com|gh-pages|github\s+actions)\b",
    ],
    "Terminal": [
        r"\b(terminal|command\s+prompt|powershell|bash|zsh|git\s+bash|cmd\.exe|wsl)\b",
        r"(ps\s+[a-z]:\\|[a-z_0-9]+@[a-z_0-9]+:[\w~]+[\$#])",
    ],
    "IDE": [
        r"\b(ide|vs\s*code|vscode|pycharm|intellij|webstorm|clion|eclipse|android\s+studio|xcode)\b",
    ],
    "LeetCode": [
        r"\b(leetcode|leetcode\.com|two\s+sum|problem\s+list|solution\s+accepted|runtime\s+beats)\b",
    ],
    "Kaggle": [
        r"\b(kaggle|kaggle\.com|kaggle\s+notebook|kaggle\s+competition)\b",
    ],
    "Docker": [
        r"\b(docker|dockerfile|docker-compose|docker\s+container|docker\s+image|docker\s+run)\b",
    ],
    "Jupyter": [
        r"\b(jupyter|jupyterlab|ipynb|jupyter\s+notebook)\b",
    ],
    # Platforms & Merchants
    "Amazon": [
        r"\b(amazon(\.in|\.com)?|amazon\s+pay|prime\s+delivery|delivered\s+by\s+amazon|amazon\s+order)\b",
    ],
    "Flipkart": [
        r"\b(flipkart(\.com)?|flipkart\s+order|supercoins|flipkart\s+assured)\b",
    ],
    "Myntra": [
        r"\b(myntra(\.com)?|myntra\s+order)\b",
    ],
    "Swiggy": [
        r"\b(swiggy|swiggy\s+instamart|swiggy\s+delivery)\b",
    ],
    "Zomato": [
        r"\b(zomato|zomato\s+gold|zomato\s+order)\b",
    ],
    "Netflix": [
        r"\b(netflix|netflix\.com)\b",
    ],
    "YouTube": [
        r"\b(youtube|youtube\.com|youtu\.be|subscribe\s+to|youtube\s+video)\b",
    ],
    "Spotify": [
        r"\b(spotify|spotify\.com|spotify\s+playlist)\b",
    ],
    # Payment Systems
    "UPI": [
        r"\b(upi(\s+(id|transaction|ref(erence)?|payment|pin))?|bhim\s+upi|vpa:?\s*[\w\.\-]+@\w+)\b",
        r"\b(gpay|phonepe|paytm\s+upi)\b",
    ],
    "Google Pay": [
        r"\b(google\s+pay|gpay|paid\s+with\s+google\s+pay)\b",
    ],
    "PhonePe": [
        r"\b(phonepe|phone\s*pe|paid\s+via\s+phonepe)\b",
    ],
    "Paytm": [
        r"\b(paytm|paytm\s+wallet|paytm\s+payments?\s+bank)\b",
    ],
    "PayPal": [
        r"\b(paypal|paypal\.com|paypal\s+balance|paid\s+with\s+paypal)\b",
    ],
}

# Content Type Rules
CONTENT_TYPE_RULES: Dict[str, List[str]] = {
    "Receipt": [
        r"\b(receipt|payment\s+receipt|e-receipt|sales\s+receipt|transaction\s+receipt)\b",
        r"\b(tax\s+invoice|customer\s+receipt|bill\s+receipt|cash\s+receipt)\b",
        r"\b(payment\s+successful|paid\s+successfully|amount\s+paid|transaction\s+successful)\b",
    ],
    "Transaction": [
        r"\b(transaction\s+(id|ref(erence)?|#|details?|history|successful|failed|completed|date))\b",
        r"\b(bank\s+reference|utr\s*(#|no|number)?\s*[:\d\w]|txn\s*(#|id|ref)?)\b",
    ],
    "Invoice": [
        r"\b(invoice\s+(#|no|number|date|due|details|summary))\b",
        r"\b(tax\s+invoice|proforma\s+invoice|commercial\s+invoice|billed\s+to|invoice\s+total)\b",
    ],
    "Order": [
        r"\b(order\s+(#|no|number|id|summary|placed|confirmed|details|tracking|history))\b",
        r"\b(thank\s+you\s+for\s+your\s+order|order\s+confirmation|view\s+order\s+details)\b",
    ],
    "Ticket": [
        r"\b(boarding\s+(pass|details)|flight\s+ticket|train\s+ticket|movie\s+ticket|bus\s+ticket|e-ticket|tatkal(\s+quota|\s+booking)?)\b",
        r"\b(pnr\s*(#|no|number)?\s*[:\d\w]|seat\s+(no|number|assignment)|gate\s+[a-z0-9]+|platform\s+\d+)\b",
        r"\b(booking\s+reference|ticket\s+(confirmation|number|details)|irctc)\b",
    ],
    "Tutorial": [
        r"\b(tutorial|how-?to\s+guide|step-by-step\s+guide|walkthrough|getting\s+started\s+with)\b",
        r"\b(learn\s+[a-zA-Z_]+\s+in\s+\d+|quickstart\s+guide|beginner('s)?\s+guide)\b",
    ],
    "Form": [
        r"\b(application\s+form|registration\s+form|contact\s+form|signup\s+form|login\s+form)\b",
        r"\b(fill\s+(out\s+)?(this|the)\s+form|form\s+submission|submit\s+form|survey\s+form)\b",
    ],
    "Job Posting": [
        r"\b(job\s+(posting|opening|description|title|alert|application|vacancy|vacancies)|we('re|\s+are)\s+hiring|careers?\s+(page|portal|opportunity|opportunities))\b",
        r"\b(apply\s+for\s+(this\s+)?(job|role|position|vacancy|opening)|apply\s+to\s+join\s+our\s+team|responsibilities\s*&?\s*requirements|qualifications\s+required)\b",
        r"\b(full-?time\s+(role|position|job)|part-?time\s+(role|position|job)|remote\s+(role|position|job)|internship\s+opening|experience:\s*\d+\+?\s*years?)\b",
    ],
    "Resume": [
        r"\b(curriculum\s+vitae|resume|work\s+experience|education\s+history|skills\s*&?\s*certifications)\b",
        r"\b(objective|summary\s+of\s+qualifications|professional\s+summary|references\s+available)\b",
    ],
    "Chat": [
        r"\b(type\s+a\s+message|message\s+history|unread\s+messages?|chat\s+history|seen\s+at)\b",
        r"\b(whatsapp|telegram|signal|discord\s+(chat|channel)|slack\s+(chat|channel)|direct\s+message)\b",
    ],
    "Email": [
        r"\b(inbox|sent\s+items|compose\s+(mail|email)|reply\s+all|forward\s+message)\b",
        r"\b(subject:\s+|from:\s+[\w\.\-]+@|to:\s+[\w\.\-]+@|attachment\s*\(\d+\))\b",
        r"\b(gmail|outlook|yahoo\s+mail|unread\s+emails?)\b",
    ],
    "Settings": [
        r"\b(settings|preferences|system\s+settings|privacy\s+settings|account\s+settings)\b",
        r"\b(configuration|device\s+preferences|advanced\s+options|notification\s+settings)\b",
    ],
    "Dashboard": [
        r"\b(analytics\s+dashboard|overview\s+dashboard|kpi|metrics\s+overview|performance\s+chart)\b",
        r"\b(monthly\s+active\s+users|conversion\s+rate|revenue\s+overview|admin\s+dashboard)\b",
    ],
    "Error Log": [
        r"\b(error\s+(code|log|message|stack|details)|uncaught\s+exception|fatal\s+error)\b",
        r"\b(internal\s+server\s+error|404\s+not\s+found|500\s+internal|traceback\s+\(most\s+recent)\b",
    ],
    "Visiting Card": [
        r"\b(visiting\s+card|business\s+card|apnacard|calling\s+card|name\s+card)\b",
    ],
}


class TagGenerator:
    """
    Deterministic rule-based tag generator for screenshots.
    Extracts tags across Domain/Activity, Entity/Application, and Content Type families.
    """

    def __init__(self):
        # Compile patterns for performance
        self._domain_compiled = {
            tag: [re.compile(p, re.IGNORECASE) for p in pats]
            for tag, pats in DOMAIN_RULES.items()
        }
        self._entity_compiled = {
            tag: [re.compile(p, re.IGNORECASE) for p in pats]
            for tag, pats in ENTITY_RULES.items()
        }
        self._content_type_compiled = {
            tag: [re.compile(p, re.IGNORECASE) for p in pats]
            for tag, pats in CONTENT_TYPE_RULES.items()
        }

    def generate_tags(
        self,
        ocr_text: str = "",
        category: Optional[str] = None,
        notes: Optional[str] = None,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Generates normalized, evidence-supported tags from OCR text and contextual signals.

        Args:
            ocr_text: Extracted text from screenshot.
            category: Canonical category prediction (e.g. 'Education', 'Shopping', etc.).
            notes: Optional human notes or dataset annotations.
            content_type: Optional pre-identified content type.
            metadata: Optional raw metadata dictionary.

        Returns:
            List of unique, normalized tag strings. Empty if insufficient evidence.
        """
        # Clean text
        raw_text = (ocr_text or "").strip()
        raw_notes = (notes or "").strip()

        # If both text and notes are essentially empty or too short to have real evidence
        combined_text = f"{raw_text} {raw_notes}".strip()
        if len(combined_text) < 4:
            return []

        # Avoid matching on pure punctuation or numbers
        if not re.search(r"[a-zA-Z]{3,}", combined_text):
            return []

        matched_tags: List[str] = []

        # 1. Match Domain/Activity tags
        for tag, patterns in self._domain_compiled.items():
            if any(p.search(combined_text) for p in patterns):
                matched_tags.append(tag)

        # 2. Match Entity/Application tags
        for tag, patterns in self._entity_compiled.items():
            if any(p.search(combined_text) for p in patterns):
                matched_tags.append(tag)

        # 3. Match Content Type tags
        for tag, patterns in self._content_type_compiled.items():
            if any(p.search(combined_text) for p in patterns):
                matched_tags.append(tag)

        # ── Contextual Disambiguation Rules ───────────────────────────────────

        # Rule A: Coding & Software Development context
        # If specific IDE or developer tools appear along with Coding, infer Software Development
        if ("VS Code" in matched_tags or "Git" in matched_tags or "GitHub" in matched_tags) and "Coding" in matched_tags:
            if "Software Development" not in matched_tags:
                matched_tags.append("Software Development")

        # If language is detected alongside code syntax, ensure Programming and Coding tags are present
        code_langs = {"Python", "JavaScript", "TypeScript", "Java", "C++", "PHP", "SQL", "HTML", "CSS"}
        detected_langs = code_langs.intersection(matched_tags)
        if detected_langs:
            # If there's an actual coding statement/syntax or developer tool, reinforce Programming
            if any(t in matched_tags for t in ("Coding", "Software Development", "IDE", "VS Code", "Terminal", "LeetCode", "Git")):
                if "Programming" not in matched_tags:
                    matched_tags.append("Programming")
                if "Coding" not in matched_tags:
                    matched_tags.append("Coding")

        # Rule B: Education + Programming disambiguation
        # If categorized as Education (e.g. lecture slide, course) with programming terms,
        # ensure Learning and Tutorial are prioritized if evidenced, but do NOT inject IDE
        if category == "Education":
            if any(t in matched_tags for t in ("Programming", "Python", "Java", "C++", "LeetCode")):
                if "Learning" not in matched_tags:
                    matched_tags.append("Learning")

        # Rule C: Shopping vs Payment disambiguation
        # A shopping order with a price should have 'Shopping' and 'Order' tags,
        # but should NOT automatically tag 'Payment' or 'Receipt' unless an explicit payment/receipt signal exists.
        if category == "Shopping" and "Shopping" in matched_tags:
            # If Payment or Receipt were matched on weak signals like "total" or "pay",
            # verify whether strong payment transaction proof exists
            has_strong_payment = bool(re.search(
                r"\b(payment\s+successful|transaction\s+(id|ref|#)|paid\s+(via|with|to)|amount\s+debited|upi\s+ref)\b",
                combined_text,
                re.IGNORECASE,
            ))
            if not has_strong_payment:
                # Remove false-positive Payment or Receipt tag
                matched_tags = [t for t in matched_tags if t not in ("Payment", "Receipt")]

        # Rule D: Payment category confirmation
        if category == "Payment":
            if any(t in matched_tags for t in ("UPI", "Google Pay", "PhonePe", "Paytm", "PayPal")) or re.search(
                r"\b(payment\s+successful|paid\s+successfully|transaction\s+successful|amount\s+transferred)\b",
                combined_text,
                re.IGNORECASE,
            ):
                if "Payment" not in matched_tags:
                    matched_tags.append("Payment")
                if "Receipt" not in matched_tags and re.search(r"\b(receipt|ref|transaction\s+id|txn)\b", combined_text, re.IGNORECASE):
                    matched_tags.append("Receipt")

        # Rule E: Never output Coding as a canonical category, only as a tag
        # (This is guaranteed by returning tags as a list, without altering category)

        # Return deterministically sorted, deduplicated tags
        return sorted(dict.fromkeys(matched_tags))

    # Backward-compatible convenience alias
    generate = generate_tags


# Global default generator instance
default_tag_generator = TagGenerator()


def generate_tags(
    ocr_text: str = "",
    category: Optional[str] = None,
    notes: Optional[str] = None,
    content_type: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Module-level convenience wrapper for default tag generator."""
    return default_tag_generator.generate_tags(
        ocr_text=ocr_text,
        category=category,
        notes=notes,
        content_type=content_type,
        metadata=metadata,
    )
