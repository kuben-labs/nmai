Tripletex — AI Accounting Agent
Build an AI agent that completes accounting tasks in Tripletex. You receive a task prompt (in one of 7 languages), use the Tripletex API to execute it, and get scored on correctness and efficiency.
How It Works

Submit your HTTPS endpoint URL on the platform
We provision a fresh Tripletex sandbox account
We send a randomly selected accounting task to your /solve endpoint
Your agent reads the prompt, optionally processes attached files (PDFs, images)
Your agent calls the Tripletex API via a proxy to complete the task
We verify the result field-by-field against expected values
Your score updates on the rolling leaderboard

Each submission gets a brand new Tripletex account — you always start from scratch.
Key Facts









Task types
30 different accounting tasks


Variants
56 per task (7 languages × 8 data sets)


Language
Prompts in Norwegian, English, Spanish, Portuguese, Nynorsk, German, French


Timeout
5 minutes per submission


API
Tripletex v2 REST API via authenticated proxy


Scoring
Field-by-field checks + efficiency bonus, best score per task kept


Score range
0.0 (failed) — up to 6.0 (perfect Tier 3 + best efficiency)


Files
Some tasks include PDF or image attachments



Quick Start

Build a /solve endpoint that accepts POST requests with a task prompt and Tripletex credentials
Use an LLM to interpret the Norwegian prompt and decide which API calls to make
Call the Tripletex API using the provided proxy URL and session token
Return {"status": "completed"} when done
Submit your endpoint URL at https://app.ainm.no/submit/tripletex

Task Categories
Your agent will encounter tasks like:

Employees — Create employees, set roles, update contact info
Customers & Products — Register customers, create products
Invoicing — Create invoices, register payments, issue credit notes
Travel Expenses — Register or delete travel expense reports
Projects — Create projects linked to customers
Corrections — Delete or reverse incorrect entries
Departments — Create departments, enable accounting modules

Tasks range from simple single-API-call operations to multi-step workflows requiring several resources to be created and linked together.