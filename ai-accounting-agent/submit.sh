#!/bin/bash

LOGFILE="/tmp/server8080.log"
LOCK=0

submit() {
  if [ "$LOCK" -eq 1 ]; then
    echo "[$(date '+%H:%M:%S')] Submit already in progress, skipping..."
    return
  fi

  LOCK=1
  while true; do
    RESPONSE=$(curl -s -X POST "https://api.ainm.no/tasks/cccccccc-cccc-cccc-cccc-cccccccccccc/submissions" \
      -H "Content-Type: application/json" \
      -H "Cookie: access_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI4ZTQ1NDRmYS00MzYyLTRjN2MtYmI5MS01MzI3MWZlNDU1ZmQiLCJlbWFpbCI6Imdvcm1lcnlrb21ib0BnbWFpbC5jb20iLCJpc19hZG1pbiI6ZmFsc2UsImV4cCI6MTc3NDU0NTA4Nn0.uVBow6pCnaKRfsb_v2Gm_2YKk6fxwZCSvK1uu4Et6Xc" \
      -d '{
        "endpoint_url": "https://5f15-2001-700-1501-d124-3847-1ea4-7abf-c2f7.ngrok-free.app/solve",
        "endpoint_api_key": null
      }')
    echo "[$(date '+%H:%M:%S')] Response: $RESPONSE"
    if echo "$RESPONSE" | grep -qi "rate exceeded"; then
      echo "Rate exceeded, retrying in 5 seconds..."
      sleep 5
    else
      break
    fi
  done
  LOCK=0
}

# kill 0 sends SIGTERM to the entire process group — kills tail and subshell too
trap "echo ''; echo 'Stopping...'; kill 0" INT TERM

echo "Starting... waiting for $LOGFILE"
while [ ! -f "$LOGFILE" ]; do
  echo "Waiting for $LOGFILE to appear..."
  sleep 2
done

echo "Log file found! Submitting once then watching for new completions..."
submit
LAST_SUBMIT=$(date +%s)

# -n 0: skip all existing content, only watch new lines
tail -n 0 -f "$LOGFILE" | while IFS= read -r line; do
  echo "[LOG] $line"

  NOW=$(date +%s)

  if echo "$line" | grep -qF '"POST /solve HTTP/1.1" 200 OK'; then
    echo "[$(date '+%H:%M:%S')] Task fully completed (200 OK) — submitting..."
    submit
    LAST_SUBMIT=$NOW

  elif [ $((NOW - LAST_SUBMIT)) -ge 180 ]; then
    echo "[$(date '+%H:%M:%S')] 3 minutes passed — submitting..."
    submit
    LAST_SUBMIT=$NOW
  fi
done