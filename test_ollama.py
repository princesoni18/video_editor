#!/usr/bin/env python3
"""Test Ollama connectivity and gemma3:4b responsiveness"""

import requests
import json
import time

OLLAMA_URL = 'http://localhost:11434/api/chat'

def test_simple():
    """Test with simple text-only request"""
    payload = {
        'model': 'gemma3:4b',
        'stream': False,
        'messages': [{'role': 'user', 'content': 'Hello, say yes'}],
        'options': {'temperature': 0.25, 'num_ctx': 8192, 'top_p': 0.9}
    }
    
    print("=" * 60)
    print("TEST 1: Simple text-only request (30s timeout)")
    print("=" * 60)
    
    start = time.time()
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=30)
        elapsed = time.time() - start
        print(f'Response received in {elapsed:.1f}s')
        print(f'Status: {resp.status_code}')
        if resp.status_code == 200:
            data = resp.json()
            msg = data.get('message', {}).get('content', '')
            print(f'Response: {msg[:100]}')
        else:
            print(f'Error: {resp.text[:200]}')
    except Exception as e:
        elapsed = time.time() - start
        print(f'FAILED after {elapsed:.1f}s: {type(e).__name__}: {e}')

def test_long_prompt():
    """Test with longer prompt (more realistic)"""
    long_prompt = """You are a video editor. Analyze this video transcript and suggest editing decisions.
    
    The video is about Python programming. Here is the transcript:
    """ + " ".join(["word"] * 100)
    
    payload = {
        'model': 'gemma3:4b',
        'stream': False,
        'messages': [{'role': 'user', 'content': long_prompt}],
        'options': {'temperature': 0.25, 'num_ctx': 8192, 'top_p': 0.9}
    }
    
    print("\n" + "=" * 60)
    print("TEST 2: Long prompt request (60s timeout)")
    print("=" * 60)
    
    start = time.time()
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
        elapsed = time.time() - start
        print(f'Response received in {elapsed:.1f}s')
        print(f'Status: {resp.status_code}')
        if resp.status_code == 200:
            data = resp.json()
            msg = data.get('message', {}).get('content', '')
            print(f'Response length: {len(msg)} chars')
            print(f'Response preview: {msg[:100]}...')
        else:
            print(f'Error: {resp.text[:200]}')
    except Exception as e:
        elapsed = time.time() - start
        print(f'FAILED after {elapsed:.1f}s: {type(e).__name__}: {e}')

if __name__ == '__main__':
    print("Testing Ollama gemma3:4b\n")
    test_simple()
    test_long_prompt()
