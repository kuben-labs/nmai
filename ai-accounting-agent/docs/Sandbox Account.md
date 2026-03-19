Tripletex — Sandbox Account
Every team gets a free Tripletex sandbox account to explore the API and web interface before submitting to the competition.
Getting Your Sandbox

Go to the Tripletex submission page on the platform
Click "Get Sandbox Account"
Your sandbox is provisioned instantly

You'll receive:

Tripletex UI URL — log in and explore the accounting interface
API base URL — call the Tripletex v2 REST API directly
Session token — authenticate your API calls

Logging Into the Web UI

Go to https://kkpqfuj-amager.tripletex.dev
Enter the email shown on your sandbox card
Click "Forgot password" to set up your Visma Connect account (first time only)
Set a password and log in

Once you've set up Visma Connect, the same credentials work for all Tripletex test accounts — including the ones created during competition submissions.
Using the API
Authenticate with Basic Auth using 0 as username and the session token as password:
import requests
 
BASE_URL = "https://kkpqfuj-amager.tripletex.dev/v2"
SESSION_TOKEN = "your-session-token-here"
 
# List employees
response = requests.get(
    f"{BASE_URL}/employee",
    auth=("0", SESSION_TOKEN),
    params={"fields": "id,firstName,lastName,email"}
)
print(response.json())
 
# Create a customer
response = requests.post(
    f"{BASE_URL}/customer",
    auth=("0", SESSION_TOKEN),
    json={
        "name": "Test Customer AS",
        "email": "test@example.com",
        "isCustomer": True,
    }
)
print(response.json())
# curl example
curl -u "0:your-session-token-here" \
  "https://kkpqfuj-amager.tripletex.dev/v2/employee?fields=id,firstName,lastName"
What You Can Do
The sandbox is a full Tripletex test environment. Use it to:

Explore the API — try creating employees, customers, invoices, and more
See the UI — understand what the accounting data looks like in the interface
Test your agent — point your /solve endpoint at the sandbox to debug
Learn the data model — see how resources relate to each other

Key Differences from Competition




Sandbox
Competition




Account
Persistent, yours to keep
Fresh account per submission


API access
Direct to Tripletex
Via authenticated proxy


Data
Accumulates over time
Starts empty each time


Scoring
None
Automated field-by-field



Tips

Create some test data manually in the UI, then query it via the API to understand the response format
Try the same operations your agent will need: creating employees, invoices, products, etc.
The sandbox token expires March 31, 2026
Each team gets one sandbox — all team members share it
