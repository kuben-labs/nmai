while true; do
  while true; do
    RESPONSE=$(curl -s -X POST "https://api.ainm.no/tasks/cccccccc-cccc-cccc-cccc-cccccccccccc/submissions" \
      -H "Content-Type: application/json" \
      -H "Cookie: access_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI4ZTQ1NDRmYS00MzYyLTRjN2MtYmI5MS01MzI3MWZlNDU1ZmQiLCJlbWFpbCI6Imdvcm1lcnlrb21ib0BnbWFpbC5jb20iLCJpc19hZG1pbiI6ZmFsc2UsImV4cCI6MTc3NDU0NTA4Nn0.uVBow6pCnaKRfsb_v2Gm_2YKk6fxwZCSvK1uu4Et6Xc" \
      -d '{
        "endpoint_url": "https://5f15-2001-700-1501-d124-3847-1ea4-7abf-c2f7.ngrok-free.app/solve",
        "endpoint_api_key": null
      }')
    echo "$RESPONSE"
    if echo "$RESPONSE" | grep -qi "rate exceeded"; then
      echo "Rate exceeded, retrying in 5 seconds..."
      sleep 5
    else
      break
    fi
  done
  sleep 180
done