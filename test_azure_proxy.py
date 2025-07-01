#!/usr/bin/env python3
"""
Test script to verify Azure OpenAI connectivity through the proxy server.

This script tests that the proxy server can successfully route requests to Azure OpenAI
using the azure/ model prefix.

Usage:
  python test_azure_proxy.py
"""

import os
import sys
import time
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_azure_through_proxy():
    """Test Azure OpenAI connection through the proxy server."""
    print("=== Testing Azure OpenAI Through Proxy ===\n")
    
    # Get Azure configuration from environment
    azure_deployment = os.environ.get("AZURE_DEPLOYMENT_NAME")
    
    if not azure_deployment:
        print("❌ AZURE_DEPLOYMENT_NAME not set in .env file")
        return False
    
    # Proxy configuration
    proxy_url = "http://localhost:8082/v1/messages"
    
    # Headers for proxy request (minimal headers needed)
    headers = {
        "content-type": "application/json",
    }
    
    # Request data using Azure model format
    request_data = {
        "model": f"azure/{azure_deployment}",
        "max_tokens": 100,
        "messages": [
            {
                "role": "user",
                "content": "Hello! Please introduce yourself in one sentence and tell me you're running on Azure."
            }
        ]
    }
    
    print(f"Testing Azure model: azure/{azure_deployment}")
    print(f"Proxy URL: {proxy_url}")
    print()
    
    try:
        # Make request to proxy
        print("Sending request to proxy...")
        start_time = time.time()
        
        response = httpx.post(
            proxy_url,
            headers=headers,
            json=request_data,
            timeout=30
        )
        
        elapsed = time.time() - start_time
        print(f"Response time: {elapsed:.2f} seconds")
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            response_json = response.json()
            print("✅ Proxy request successful!")
            
            # Extract content from response
            if "content" in response_json and response_json["content"]:
                content = response_json["content"]
                if isinstance(content, list) and len(content) > 0:
                    text_content = ""
                    for item in content:
                        if item.get("type") == "text":
                            text_content = item.get("text", "")
                            break
                    
                    if text_content:
                        print(f"Response: {text_content}")
                    else:
                        print("No text content found in response")
                else:
                    print("Empty content in response")
            
            # Print usage info if available
            if "usage" in response_json:
                usage = response_json["usage"]
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                print(f"Tokens used: {input_tokens + output_tokens} (input: {input_tokens}, output: {output_tokens})")
            
            return True
        else:
            print(f"❌ Proxy request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except httpx.ConnectError:
        print("❌ Could not connect to proxy server at http://localhost:8082")
        print("Make sure the server is running with: uv run uvicorn server:app --host 0.0.0.0 --port 8082 --reload")
        return False
    except Exception as e:
        print(f"❌ Error testing proxy: {str(e)}")
        return False

def main():
    """Main function to run the proxy test."""
    success = test_azure_through_proxy()
    
    if success:
        print("\n🎉 Azure OpenAI proxy test passed!")
        sys.exit(0)
    else:
        print("\n❌ Azure OpenAI proxy test failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()