import json
import os
from strands import Agent
from strands_tools import current_time, python_repl
from strands.models import BedrockModel
import boto3
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define a system prompt with strict JSON output
SYSTEM_PROMPT = """
You are an AI assistant that optimizes Medium post publishing times. Your response must be a valid JSON object with no text outside it. Follow these steps:
1. Use the current_time tool to get the current time in US/Pacific timezone.
2. Use the python_repl tool to analyze engagement data from '$PATH/engagement_data.csv' with columns: 'timestamp' (ISO 8601 in US/Pacific), 'claps' (int), 'comments' (int).
   - Convert any NumPy types (e.g., np.float64) to native Python types (e.g., float) before returning results.
   - Handle errors like file not found, permission denied, or malformed data, and return an error JSON.
3. Group the data by hour and calculate the average engagement (claps + comments) for each hour.
4. Identify the hour with the highest average engagement.
5. Compare the current hour to the best hour and recommend posting now (if matching) or scheduling for the best hour today (May 30, 2025).
6. Return a JSON object with the following structure:
{
  "status": "success" or "error",
  "best_hour": "HH:MM" (if success),
  "average_engagement": float (if success),
  "current_time": "YYYY-MM-DDTHH:MM:SS-07:00" (if success),
  "recommendation": "string" (if success),
  "reasoning": "string" (if success),
  "message": "string" (if error)
}
Example success output:
{
  "status": "success",
  "best_hour": "20:00",
  "average_engagement": 170.0,
  "current_time": "2025-05-30T07:42:00-07:00",
  "recommendation": "Schedule your post for 8 PM today, May 30, 2025.",
  "reasoning": "The hour 20:00 has the highest average engagement (150 claps + 20 comments)."
}
Example error output:
{
  "status": "error",
  "message": "CSV file '$PATH/engagement_data.csv' not found or not readable."
}
"""

# Define the tools
tools = [current_time, python_repl]

# Create a boto3 session using environment variables
boto_session = boto3.Session(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name="us-east-1"
)

# Configure the Bedrock model with non-streaming
model = BedrockModel(
    boto_session=boto_session,
    model_id=os.getenv("modelid"),
    max_tokens=1000,
    streaming=False
)

# Create the agent
agent = Agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=tools
)

# Run the agent
response = agent("""
Maximize engagement for my Medium post.
Analyze '$PATH/engagement_data.csv' and suggest the best time to post today (May 30, 2025).
Return the result as a JSON object.
""")

# Extract and print the JSON response
if hasattr(response, 'messages') and response.messages:
    content = response.messages[-1].get('content', '{}')
    try:
        result = json.loads(content)
        # Convert np.float64 to float
        if isinstance(result.get('average_engagement'), float):
            result['average_engagement'] = float(result['average_engagement'])
        print(json.dumps(result))
    except json.JSONDecodeError:
        print(json.dumps({"status": "error", "message": "Failed to parse JSON from content", "raw_content": content}))
else:
    print(json.dumps({"status": "error", "message": f"Unexpected response type: {type(response)}", "raw_response": str(response)}))
