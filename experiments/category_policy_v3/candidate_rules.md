# POLICY V3 CANDIDATE RULES SPECIFICATION
=========================================

| Rule ID | Category | Trigger Evidence | Negative Exclusions | Confusable Pairs | Expected Rationale |
|---|---|---|---|---|---|
| **V3_Rule1_Banking_Dashboard** | General | `net worth`, `account balances`, `total assets`, `operating balance` | `payment successful`, `transaction id`, `paid to` | General vs Payment | Bank account balance summaries & net worth trackers are General UI overviews, not P2P payment receipts. |
| **V3_Rule2_Dev_Admin_Tools** | Work & Career | `xampp control panel`, `phpmyadmin`, `localhost/`, `mysql error` | None | Work & Career vs General / Entertainment | Server management and database admin portals are developer workspace tools. |
| **V3_Rule3_Payment_Forms** | Payment | `payment submitted`, `depositing amounts`, `recipient name`, `transfer amount` | None | Payment vs Communication | Deposit forms and transaction submissions are Payment receipts. |
| **V3_Rule4_Retail_Collections** | Shopping | `urban clothing`, `shoes accessories`, `new season sale` | None | Shopping vs Entertainment | Fashion retail collection listings are Shopping products. |
