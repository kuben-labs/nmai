"""Twitter MCP Server using Twitter API v2

This server provides tools to interact with Twitter (X) API v2 including:
- Posting tweets
- Searching tweets
- Getting user information
"""

import os
from typing import Optional
from fastmcp import FastMCP
from pytwitter import Api
from pytwitter.error import PyTwitterError
from dotenv import load_dotenv
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP(
    name="twitter-mcp",
    version="0.1.0"
)

# Initialize Twitter API client with OAuth 1.0a
def get_twitter_client() -> Api:
    """Initialize and return Twitter API client with OAuth 1.0a credentials"""
    consumer_key = os.getenv("API_KEY")
    consumer_secret = os.getenv("API_SECRET_KEY")
    access_token = os.getenv("ACCESS_TOKEN")
    access_secret = os.getenv("ACCESS_TOKEN_SECRET")
    
    # Debug logging (remove in production)
    print(f"DEBUG - API_KEY present: {bool(consumer_key)}")
    print(f"DEBUG - API_SECRET_KEY present: {bool(consumer_secret)}")
    print(f"DEBUG - ACCESS_TOKEN present: {bool(access_token)}")
    print(f"DEBUG - ACCESS_TOKEN_SECRET present: {bool(access_secret)}")
    
    if not all([consumer_key, consumer_secret, access_token, access_secret]):
        raise ValueError(
            "Missing required environment variables. Please set: "
            "API_KEY, API_SECRET_KEY, ACCESS_TOKEN, ACCESS_TOKEN_SECRET"
        )
    
    return Api(
        consumer_key=consumer_key,
        consumer_secret=consumer_secret,
        access_token=access_token,
        access_secret=access_secret
    )


@mcp.tool()
def post_tweet(text: str, reply_to_tweet_id: str = "") -> str:
    """Post a tweet to Twitter

    Args:
        text: The content of the tweet (max 280 characters)
        reply_to_tweet_id: Optional tweet ID to reply to (empty string for no reply)

    Returns:
        Success message with tweet ID or error message
    """
    try:
        api = get_twitter_client()
        
        # Validate tweet length
        if len(text) > 280:
            return f"Error: Tweet text exceeds 280 characters (current: {len(text)})"
        
        if len(text) == 0:
            return "Error: Tweet text cannot be empty"
        
        # Create tweet with optional reply
        if reply_to_tweet_id:
            response = api.create_tweet(
                text=text,
                reply_in_reply_to_tweet_id=reply_to_tweet_id
            )
            # Response can be either Response object with .data or direct Tweet object
            tweet_id = response.data.id if hasattr(response, 'data') else response.id
            tweet_text = response.data.text if hasattr(response, 'data') else response.text
            return f"Tweet posted successfully as reply! Tweet ID: {tweet_id}\nText: {tweet_text}"
        else:
            response = api.create_tweet(text=text)
            # Response can be either Response object with .data or direct Tweet object
            tweet_id = response.data.id if hasattr(response, 'data') else response.id
            tweet_text = response.data.text if hasattr(response, 'data') else response.text
            return f"Tweet posted successfully! Tweet ID: {tweet_id}\nText: {tweet_text}"
            
    except PyTwitterError as e:
        error_detail = str(e)
        if "401" in error_detail or "Unauthorized" in error_detail:
            return (
                "Twitter API error: 401 Unauthorized\n\n"
                "This usually means:\n"
                "1. Your access tokens are invalid or expired\n"
                "2. You changed app permissions but didn't regenerate tokens\n"
                "3. Your app doesn't have Read+Write permissions\n\n"
                "To fix:\n"
                "- Go to https://developer.twitter.com/en/portal/dashboard\n"
                "- Navigate to your app settings\n"
                "- Ensure 'Read and Write' permissions are enabled\n"
                "- Regenerate your Access Token and Secret\n"
                "- Update your environment variables with the new tokens"
            )
        return f"Twitter API error: {error_detail}"
    except ValueError as e:
        return f"Configuration error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


@mcp.tool()
def search_tweets(query: str, max_results: int = 10) -> str:
    """Search for recent tweets on Twitter

    Args:
        query: Search query string (supports Twitter search operators)
        max_results: Maximum number of tweets to return (10-100, default 10)

    Returns:
        Formatted string with tweet results or error message
    """
    try:
        api = get_twitter_client()
        
        # Validate max_results
        if max_results < 10:
            max_results = 10
        elif max_results > 100:
            max_results = 100
        
        # Search tweets with expanded fields
        response = api.search_tweets(
            query=query,
            max_results=max_results,
            tweet_fields=["created_at", "public_metrics", "author_id"],
            expansions=["author_id"],
            user_fields=["username", "name", "verified"]
        )
        
        if not response.data:
            return f"No tweets found for query: {query}"
        
        # Format results
        results = [f"Found {len(response.data)} tweets for query: '{query}'\n"]
        
        # Create user lookup dictionary
        users_dict = {}
        if hasattr(response, 'includes') and hasattr(response.includes, 'users'):
            users_dict = {user.id: user for user in response.includes.users}
        
        for i, tweet in enumerate(response.data, 1):
            # Get author info
            author = users_dict.get(tweet.author_id)
            author_name = f"@{author.username}" if author else f"User {tweet.author_id}"
            
            # Get metrics
            likes = tweet.public_metrics.like_count if hasattr(tweet, 'public_metrics') else 0
            retweets = tweet.public_metrics.retweet_count if hasattr(tweet, 'public_metrics') else 0
            
            results.append(
                f"\n{i}. {author_name} (ID: {tweet.id})\n"
                f"   {tweet.text}\n"
                f"   ❤️  {likes} likes | 🔁 {retweets} retweets"
            )
        
        return "\n".join(results)
        
    except PyTwitterError as e:
        error_detail = str(e)
        if "429" in error_detail or "Too Many Requests" in error_detail:
            return (
                "Twitter API error: 429 Rate Limit Exceeded\n\n"
                "You've hit the Twitter API rate limit for search.\n"
                "Rate limits for Twitter API v2 (Free tier):\n"
                "- Recent search: 180 requests per 15 minutes\n"
                "- You may need to wait before searching again\n\n"
                "If this happens frequently, consider:\n"
                "1. Reducing search frequency\n"
                "2. Upgrading to a higher Twitter API tier\n"
                "3. Implementing caching for repeated searches"
            )
        return f"Twitter API error: {error_detail}"
    except ValueError as e:
        return f"Configuration error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


@mcp.tool()
def get_user_info(username: str) -> str:
    """Get information about a Twitter user

    Args:
        username: Twitter username (without @)

    Returns:
        User information or error message
    """
    try:
        api = get_twitter_client()
        
        # Remove @ if present
        username = username.lstrip('@')
        
        # Get user information
        response = api.get_user(
            username=username,
            user_fields=["created_at", "description", "public_metrics", "verified"]
        )
        
        if not response.data:
            return f"User not found: @{username}"
        
        user = response.data
        metrics = user.public_metrics if hasattr(user, 'public_metrics') else None
        
        result = [
            f"User: @{user.username}",
            f"Name: {user.name}",
            f"ID: {user.id}",
        ]
        
        if hasattr(user, 'description') and user.description:
            result.append(f"Bio: {user.description}")
        
        if hasattr(user, 'verified'):
            result.append(f"Verified: {'✓' if user.verified else '✗'}")
        
        if metrics:
            result.extend([
                f"\nFollowers: {metrics.followers_count:,}",
                f"Following: {metrics.following_count:,}",
                f"Tweets: {metrics.tweet_count:,}"
            ])
        
        if hasattr(user, 'created_at'):
            result.append(f"Joined: {user.created_at}")
        
        return "\n".join(result)
        
    except PyTwitterError as e:
        return f"Twitter API error: {str(e)}"
    except ValueError as e:
        return f"Configuration error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


@mcp.tool()
def delete_tweet(tweet_id: str) -> str:
    """Delete a tweet

    Args:
        tweet_id: ID of the tweet to delete

    Returns:
        Success or error message
    """
    try:
        api = get_twitter_client()
        
        response = api.delete_tweet(tweet_id=tweet_id)
        
        if response.get('data', {}).get('deleted'):
            return f"Tweet {tweet_id} deleted successfully!"
        else:
            return f"Failed to delete tweet {tweet_id}"
            
    except PyTwitterError as e:
        return f"Twitter API error: {str(e)}"
    except ValueError as e:
        return f"Configuration error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


if __name__ == "__main__":
    # Run as HTTP server for MCP
    mcp.run(transport="http", host="0.0.0.0", port=8083)
