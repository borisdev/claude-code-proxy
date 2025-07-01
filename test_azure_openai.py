#!/usr/bin/env python3
"""
Simple test to verify Azure OpenAI connectivity using LiteLLM.

This test directly uses LiteLLM to connect to Azure OpenAI, bypassing the proxy server.
It validates that the Azure OpenAI configuration in .env works correctly.

Usage:
  python test_azure_openai.py
"""

import os
import sys

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def test_azure_openai_direct():
    """Test direct connection to Azure OpenAI using LiteLLM."""
    print("=== Testing Azure OpenAI Direct Connection ===\n")

    # Get Azure configuration from environment
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    azure_api_version = os.environ.get("AZURE_API_VERSION")
    azure_deployment = os.environ.get("AZURE_DEPLOYMENT_NAME")

    # Check if Azure configuration is available
    if not azure_endpoint or not azure_api_key:
        print("❌ Azure OpenAI configuration missing!")
        print("Required environment variables:")
        print("  - AZURE_OPENAI_ENDPOINT")
        print("  - AZURE_OPENAI_API_KEY")
        print("  - AZURE_API_VERSION (optional)")
        print("  - AZURE_DEPLOYMENT_NAME (optional)")
        return False

    print(f"Azure Configuration:")
    print(f"  Endpoint: {azure_endpoint}")
    print(f"  API Version: {azure_api_version or 'default'}")
    print(f"  Deployment: {azure_deployment or 'default'}")
    print()

    try:
        # Import LiteLLM
        import litellm

        # Construct the Azure model string for LiteLLM
        # Format: azure/{deployment_name}
        model = f"azure/{azure_deployment}" if azure_deployment else "azure/gpt-4"

        print(f"Testing LiteLLM connection with model: {model}")

        # Prepare the completion request
        messages = [
            {
                "role": "user",
                "content": "Hello! Please introduce yourself in one sentence.",
            }
        ]

        # Make the request to Azure OpenAI via LiteLLM
        response = litellm.completion(
            model=model,
            messages=messages,
            api_key=azure_api_key,
            api_base=azure_endpoint,
            api_version=azure_api_version,
            max_tokens=100,
        )

        # Check if we got a valid response
        if response and hasattr(response, "choices") and len(response.choices) > 0:
            message = response.choices[0].message
            content = message.content if hasattr(message, "content") else str(message)

            print("✅ Azure OpenAI connection successful!")
            print(f"Response: {content}")

            # Print usage info if available
            if hasattr(response, "usage"):
                usage = response.usage
                print(f"Tokens used: {getattr(usage, 'total_tokens', 'unknown')}")

            return True
        else:
            print("❌ Azure OpenAI connection failed: Empty or invalid response")
            return False

    except ImportError:
        print("❌ LiteLLM not installed. Please install with: pip install litellm")
        return False
    except Exception as e:
        print(f"❌ Azure OpenAI connection failed: {str(e)}")

        # Print additional error details if available
        if hasattr(e, "status_code"):
            print(f"Status code: {e.status_code}")
        if hasattr(e, "response"):
            print(f"Response: {e.response}")
        if hasattr(e, "message"):
            print(f"Message: {e.message}")

        return False


def main():
    """Main function to run the Azure OpenAI test."""
    success = test_azure_openai_direct()

    if success:
        print("\n🎉 Azure OpenAI test passed!")
        sys.exit(0)
    else:
        print("\n❌ Azure OpenAI test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
